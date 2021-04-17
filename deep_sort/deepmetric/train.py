import argparse
import os
import time

import matplotlib.pyplot as plt
import paddle as torch
import paddle.vision as torchvision
from paddle.fluid.reader import DataLoader
from paddle.optimizer.lr import StepDecay
from paddle.static import InputSpec
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
import paddle.fluid as fluid
from deep_sort.deepmetric.dataset import ReIDDataset
from deep_sort.deepmetric.model import Net
from deep_sort.deepmetric.loss import ArcMarginLoss

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='./data/archive/Market-1501-v15.09.15', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if not args.no_cuda else "cpu"

# data loading
root = args.data_dir
train_dir = os.path.join(root, "bounding_box_train")
test_dir = os.path.join(root, "bounding_box_test")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainloader = DataLoader(
    ReIDDataset(train_dir, transform=transform_train),
    batch_size=64, shuffle=True
)
testloader = DataLoader(
    ReIDDataset(test_dir, transform=transform_test),
    batch_size=64, shuffle=True
)
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))

# net definition
start_epoch = 0
#net = ResNet(depth=18,num_classes=-1,block=BottleneckBlock)
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# loss and optimizer
#criterion = torch.nn.CrossEntropyLoss()
criterion = ArcMarginLoss(class_dim=num_classes)
sch = StepDecay(learning_rate=args.lr, step_size=20, verbose=1, gamma=0.100)
optimizer = torch.optimizer.Momentum(parameters=net.parameters(), learning_rate=sch,momentum=0.9, weight_decay=5e-4)
best_acc = 0.
metric = torch.metric.Accuracy()


# train function for each epoch
def train(epoch):
    metric.reset()
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):

        outputs = net(inputs)
        loss,outputs = criterion(outputs, labels)
        # backward
        optimizer.clear_grad()  # zero_grad()
        loss.backward()
        optimizer.step()
        cc = metric.compute(outputs, labels)
        metric.update(cc)
        res = metric.accumulate()
        # accumurating
        training_loss += loss.mean().numpy().item()
        train_loss += loss.mean().numpy().item()
        correct += 0  # fluid.layers.equal(outputs.max(axis=1)[1],torch.cast(labels,'float32')).sum().numpy().item()
        total += 1  # labels.size(0)

        # print 
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total, res
            ))
            training_loss = 0.
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total


@torch.no_grad()
def test(epoch):
    metric.reset()
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    for idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs, labels
        outputs = net(inputs)
        loss,outputs = criterion(outputs, labels)

        cc = metric.compute(outputs, labels)
        metric.update(cc)
        res = metric.accumulate()
        test_loss += loss.mean().numpy().item()
        correct += 0  # outputs.max(dim=1)[1].eq(labels).sum().numpy().item()
        total += 1  # labels.size(0)

    print("Testing ...")
    end = time.time()
    print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}".format(
        100. * (idx + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total, res
    ))

    print("Saving parameters to checkpoint/ckpt.t7")
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.jit.save(net, './checkpoint/net', input_spec=[InputSpec(shape=[1, 3, 128, 64], dtype='float32')])

    return test_loss / len(testloader), 1. - correct / total


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range(start_epoch, start_epoch + 40):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        sch.step()


if __name__ == '__main__':
    main()

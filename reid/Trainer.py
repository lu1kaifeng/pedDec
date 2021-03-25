import paddle
from paddle import fluid
from paddle.fluid.reader import DataLoader
from paddle.optimizer.lr import StepDecay
from paddle.vision import transforms

from reid.dataset import ReIDDataset
from reid.modeling import PCB


class Trainer:
    def __init__(self, pretrained=None, last_epoch=-1):
        self._ds_transforms = transforms.Compose([transforms.Resize((384, 192), interpolation='bicubic'),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [
                                                      0.229, 0.224, 0.225])
                                                  ])
        self.model = PCB(1500)
        # model = paddle.Model(PCB(1500))
        self.scheduler = StepDecay(learning_rate=0.1, step_size=40, verbose=1, last_epoch=last_epoch, gamma=0.130)
        self.train_loader = None
        self.test_loader = None
        self.last_epoch = last_epoch
        # scheduler = ReduceOnPlateau(learning_rate=0.1,verbose=1,patience=40)
        self.sgd = paddle.optimizer.SGD(parameters=self.model.parameters(), weight_decay=5e-4,
                                        learning_rate=self.scheduler)
        self.metric = paddle.metric.Accuracy()
        self.ce_loss = paddle.nn.CrossEntropyLoss()
        if pretrained is not None:
            params_dict, opt_dict = fluid.load_dygraph(pretrained)
            self.model.load_dict(params_dict)
            self.sgd.set_state_dict(opt_dict)

    def train(self):
        if self.train_loader is None:
            train_ds = ReIDDataset('./data/archive/Market-1501-v15.09.15/bounding_box_train',
                                   transform=self._ds_transforms)
            self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        if self.test_loader is None:
            test_ds = ReIDDataset('./data/archive/Market-1501-v15.09.15/bounding_box_test',
                                  transform=self._ds_transforms)
            self.test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)
        for epoch in range(self.last_epoch + 1, 100):
            self.metric.reset()
            for batch in self.train_loader:
                info = ''
                out = self.model(batch[0])
                loss = self.ce_loss(out, batch[1])
                loss.backward()
                self.sgd.step()
                self.sgd.clear_gradients()
                info += 'epoch: ' + str(epoch)
                info += ' loss:' + str(loss.numpy().item())
                correct = self.metric.compute(out, batch[1])
                self.metric.update(correct)
                res = self.metric.accumulate()
                info += 'acc: ' + str(res)
                print(info)

            if (epoch + 1) % 4 == 0:
                fluid.save_dygraph(self.model.state_dict(), "./PCB_1/" + str(epoch))
                fluid.save_dygraph(self.sgd.state_dict(), "./PCB_1/" + str(epoch))
                print('model saved')
            self.scheduler.step()

    @paddle.no_grad()
    def eval(self):
        if self.test_loader is None:
            test_ds = ReIDDataset('./data/archive/Market-1501-v15.09.15/bounding_box_test',
                                  transform=self._ds_transforms)
            self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
        self.metric.reset()
        for batch in self.test_loader:
            info = ''
            out = self.model(batch[0])
            info += 'eval: '
            info += 'acc: ' + str(paddle.metric.accuracy(out[0], batch[1][0]).numpy().item())
            print(info)


if __name__ == '__main__':
    t = Trainer(pretrained='./PCB_1/59')
    t.eval()

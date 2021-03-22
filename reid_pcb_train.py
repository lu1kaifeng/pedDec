import paddle
from paddle.fluid.clip import GradientClipByValue
from paddle.fluid.reader import DataLoader
from paddle.vision import transforms

from reid.dataset import ReIDDataset
from reid.modeling import PCB
from paddle.optimizer.lr import StepDecay,ReduceOnPlateau
eval_transforms = transforms.Compose([transforms.Resize((384, 192),interpolation='bicubic'),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [
                                0.229, 0.224, 0.225])
                            ])
train_ds = ReIDDataset('./data/archive/Market-1501-v15.09.15/bounding_box_train',transform=eval_transforms)
test_ds = ReIDDataset('./data/archive/Market-1501-v15.09.15/bounding_box_test',transform=eval_transforms)
model = PCB(1500)
#model = paddle.Model(PCB(1500))
scheduler = StepDecay(learning_rate=0.1,step_size=40,verbose=1,gamma=0.130)
train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
#scheduler = ReduceOnPlateau(learning_rate=0.1,verbose=1,patience=40)
sgd = paddle.optimizer.SGD(parameters=model.parameters(),weight_decay=5e-4,learning_rate=scheduler)
m = paddle.metric.Accuracy()
ce_loss = paddle.nn.CrossEntropyLoss()
for epoch in range(100):
    m.reset()
    for batch in train_loader:
        out = model(batch[0])
        loss = ce_loss(out,batch[1])
        loss.backward()
        sgd.step()
        sgd.clear_gradients()
        print('epoch: '+str(epoch),end=' ')
        print(loss[0].item,end=' ')
        correct = m.compute(out, batch[1])
        m.update(correct)
        res = m.accumulate()
        print(res)
    if (epoch + 1) % 4 == 0:
        paddle.save(model.state_dict(), "./PCB/"+str(epoch)+"/pcb_net.pdparams")
        paddle.save(sgd.state_dict(), "./PCB/"+str(epoch)+"/sgd.pdopt")
        print('epoch '+ str(epoch)+' saved')
    scheduler.step()    # If you update learning rate each step
              # scheduler.step(loss)
# model.prepare(optimizer=paddle.optimizer.SGD(parameters=model.parameters(),weight_decay=5e-4,learning_rate=scheduler),
#               loss=paddle.nn.CrossEntropyLoss(),
#               metrics=paddle.metric.Accuracy())
# model.fit(train_ds,
#           epochs=60,
#           batch_size=32,
#           verbose=2)
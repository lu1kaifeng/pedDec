import numpy as np
import paddle as torch
from paddle.fluid.reader import DataLoader

from deep_sort.deepaddle.dataset import ReIDDataset
from deep_sort.deepaddle.model import Net

model_path = 'checkpoint/net'
model = torch.jit.load(model_path)
net = Net(reid=True)
net.set_state_dict(model.state_dict())

import paddle.vision as torchvision
import os

root = 'data/archive/Market-1501-v15.09.15'
train_dir = os.path.join(root, "bounding_box_train")
test_dir = os.path.join(root, "bounding_box_test")
size = (64, 128)
norm = torchvision.transforms.Compose([
    # torchvision.transforms.ToTensor(data_format='CHW'),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], data_format='HWC'),
])
trainloader = DataLoader(
    ReIDDataset(train_dir, transform=norm),
    batch_size=1, shuffle=True
)
for ip in trainloader:
    # out_dygraph, static_layer = TracedLayer.trace(net, inputs=np.moveaxis(ip[0], (3), (1)))
    input_spec = torch.static.InputSpec(shape=[1, 3, 128, 64], dtype='float32')
    net = torch.jit.to_static(net, input_spec=[input_spec])
    net(np.moveaxis(ip[0], (3), (1)))
    torch.jit.save(model, './checkpoint_static/net', input_spec=[input_spec])
    # print(static_layer(inputs=np.moveaxis(ip[0], (3), (1))))
    # static_layer.save_inference_model('./checkpoint_static/net', feed=[0], fetch=[0])
    break

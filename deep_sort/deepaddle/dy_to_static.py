import paddle as torch
import numpy as np
from paddle.fluid.reader import DataLoader

from deep_sort.deepaddle.dataset import ReIDDataset
from deep_sort.deepaddle.model import  Net
model_path = 'checkpoint/net'
model = torch.jit.load(model_path)
net = Net()
net.set_state_dict(model.state_dict())

from paddle.jit import TracedLayer
import paddle.vision as torchvision
import os
root = 'data/archive/Market-1501-v15.09.15'
train_dir = os.path.join(root, "bounding_box_train")
test_dir = os.path.join(root, "bounding_box_test")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainloader = DataLoader(
    ReIDDataset(train_dir, transform=transform_train),
    batch_size=1, shuffle=True
)
for ip in trainloader:
    out_dygraph, static_layer = TracedLayer.trace(net,inputs=ip[0])
    static_layer.save_inference_model('./checkpoint_static/net', feed=[0], fetch=[0])
    break
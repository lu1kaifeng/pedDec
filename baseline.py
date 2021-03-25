base = './data/'
import os

import paddlex as pdx
from paddlex.det import transforms

train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=512, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=512, interp='CUBIC'),
    transforms.Normalize(),
])
train_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=os.path.join(base, 'train.txt'),
    label_list='./data/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=os.path.join(base, 'valid.txt'),
    transforms=eval_transforms,
    label_list='./data/labels.txt')

num_classes = len(train_dataset.labels) + 1
print('class num:', num_classes)
model = pdx.det.YOLOv3(
    num_classes=num_classes,
    backbone='MobileNetV3_large'
)
model.train(
    num_epochs=60,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.00025,
    lr_decay_epochs=[20, 40],
    save_interval_epochs=4,
    log_interval_steps=100,
    save_dir='./YOLOv3')

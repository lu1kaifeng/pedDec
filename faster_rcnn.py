base = './data/'
import os

import paddlex as pdx
from paddlex.det import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32), ])
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
model = pdx.det.FasterRCNN(
    num_classes=num_classes
)
model.train(
    num_epochs=18,
    train_dataset=train_dataset,
    train_batch_size=2,
    save_interval_epochs=18,
    eval_dataset=eval_dataset,
    learning_rate=0.0025,
    lr_decay_epochs=[8, 11, 13, 15, 17],
    save_dir='./FasterRCNN', resume_checkpoint='./FasterRCNN/epoch_12')

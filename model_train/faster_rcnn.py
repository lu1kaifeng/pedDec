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
    file_list=os.path.join(base, 'train_all_1.txt'),
    label_list='./data/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=os.path.join(base, 'valid_all_1.txt'),
    transforms=eval_transforms,
    label_list='./data/labels.txt')

num_classes = len(train_dataset.labels) + 1
print('class num:', num_classes)
model = pdx.det.FasterRCNN(
    num_classes=num_classes
)
model.train(
    num_epochs=50,
    train_dataset=train_dataset,
    train_batch_size=2,
    save_interval_epochs=1,
    eval_dataset=eval_dataset,
    learning_rate=0.0025,
    lr_decay_epochs=[8, 11,25],
    save_dir='./models/det/FasterRCNN', resume_checkpoint='./models/det/FasterRCNN/epoch_30',use_vdl=True)

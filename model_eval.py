import cv2
import time
import os
import paddlex as pdx
from paddlex.det import transforms
import numpy as np
import matplotlib.pyplot as plt
rcnn_eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32),])
rcnn_model = pdx.load_model('./FasterRCNN/best_model')

yolo_eval_transforms = transforms.Compose([
    transforms.Resize(target_size=512, interp='CUBIC'),
    transforms.Normalize(),
])
yolo_model = pdx.load_model('./YOLOv3/best_model')
print(rcnn_model.evaluate(pdx.datasets.VOCDetection(
    data_dir='./data',
    file_list=os.path.join('./data', 'valid.txt'),
    transforms=rcnn_eval_transforms,
    label_list='./data/labels.txt'),16))
print(yolo_model.evaluate(pdx.datasets.VOCDetection(
    data_dir='./data',
    file_list=os.path.join('./data', 'valid.txt'),
    transforms=yolo_eval_transforms,
    label_list='./data/labels.txt'),16))
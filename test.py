import time

import cv2
import numpy as np
import paddle
from tqdm import tqdm

from deep_sort import DeepSort
from evaluation.CsvEvalWriter import CsvEvalWriter

base = './data/'
import os
import csv
bboxes = {1:[]}
with open('MOT20-01.txt', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if int(row[0]) not in bboxes:
            bboxes[int(row[0])] = []
        bboxes[int(row[0])].append(row[1:])

import paddlex as pdx
from paddlex.det import transforms
INTERACTIVE=True
WRITE_CSV=False
USE_YOLO=False
if USE_YOLO:
    eval_transforms = transforms.Compose([
        transforms.Resize(target_size=512, interp='CUBIC'),
        transforms.Normalize(),
    ])

else:
    eval_transforms = transforms.Compose([
        transforms.Normalize(),
        transforms.ResizeByShort(short_size=800, max_size=1333),
        transforms.Padding(coarsest_stride=32), ])

loop_gen = ( ( pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=k,
    transforms=eval_transforms,
    label_list='./data/labels.txt'),v) for k,v in [
('data/MOT20/train/MOT20-01/manifest.txt','MOT20-01.txt'),
                            ('data/MOT20/train/MOT20-02/manifest.txt','MOT20-02.txt'),
                            ('data/MOT20/train/MOT20-03/manifest.txt','MOT20-03.txt'),
    ('data/MOT20/train/MOT20-05/manifest.txt', 'MOT20-05.txt')

                            ] )


ii=1
for ds,txt in loop_gen:
    for i in tqdm(ds.file_list):
        image_name = i[0]
        start = time.time()
        # print('infer time:{:.6f}s'.format(time.time()-start))
        # print('detected num:', len(result))
       # paddle.disable_static()
        im = cv2.imread(image_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        threshold = 0.2
        for value in bboxes[ii]:
            track,x, y, w, h,conf,e,ee,eee = value
            track=int(track)
            x=int(float(x))
            y=int(float(y))
            w=int(float(w))
            h=int(float(h))
            conf=int(float(conf))
            if INTERACTIVE:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(im, '{:d} {:d}'.format(track, track),
                            (x, y), font, 0.5, (255, 0, 0), thickness=2)
        if INTERACTIVE:
            cv2.imshow('result', im)
            cv2.waitKey(0)
        ii+=1
        #paddle.enable_static()

        # plt.figure(figsize=(15,12))
        # plt.imshow(im[:, :, [2,1,0]])
        # plt.show()

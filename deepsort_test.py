import time

import cv2
import numpy as np
import paddle
from tqdm import tqdm

from deep_sort import DeepSort

sort = DeepSort('checkpoint_static/net', n_init=2)
base = './data/'
import os

import paddlex as pdx
from paddlex.det import transforms

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=512, interp='CUBIC'),
    transforms.Normalize(),
])
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=base,
    file_list=os.path.join(base, 'valid.txt'),
    transforms=eval_transforms,
    label_list='./data/labels.txt')

model = pdx.load_model('./YOLOv3/best_model')

for i in tqdm(eval_dataset.file_list):
    image_name = i[0]
    start = time.time()
    result = model.predict(image_name)
    # print('infer time:{:.6f}s'.format(time.time()-start))
    # print('detected num:', len(result))
   # paddle.disable_static()
    im = cv2.imread(image_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.2
    result = list(filter(lambda x: x['score'] > threshold, result))
    bboxes = np.array(list(map(lambda v: np.array(v['bbox']), result)))
    confidence = list(map(lambda v: v['score'], result))
    track = sort.update(bboxes, confidence, im)
    for value in result:
        xmin, ymin, w, h = np.array(value['bbox']).astype(np.int)
        cls = value['category']
        score = value['score']
        cv2.rectangle(im, (xmin, ymin), (xmin + w, ymin + h), (255, 0, 0), 4)
        cv2.putText(im, '{:s} {:.3f}'.format(cls, score),
                    (xmin, ymin), font, 0.5, (0, 225, 0), thickness=1)
    for value in track:
        x, y, w, h, track = value
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(im, '{:d} {:d}'.format(track, track),
                    (x, y), font, 0.5, (255, 0, 0), thickness=2)

    # cv2.imshow('result', im)
    #paddle.enable_static()
    # cv2.waitKey(0)
    # plt.figure(figsize=(15,12))
    # plt.imshow(im[:, :, [2,1,0]])
    # plt.show()

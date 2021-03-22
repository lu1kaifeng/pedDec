import cv2
import time
import paddlex as pdx
from paddlex.det import transforms
import numpy as np
import matplotlib.pyplot as plt
eval_transforms = transforms.Compose([
    transforms.Resize(target_size=512, interp='CUBIC'),
    transforms.Normalize(),
])
model = pdx.load_model('./YOLOv3/best_model')
image_name = './test.jpg'
start = time.time()
result = model.predict(image_name, eval_transforms)
print('infer time:{:.6f}s'.format(time.time()-start))
print('detected num:', len(result))

im = cv2.imread(image_name)
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.1

for value in result:
    xmin, ymin, w, h = np.array(value['bbox']).astype(np.int)
    cls = value['category']
    score = value['score']
    if score < threshold:
        continue
    cv2.rectangle(im, (xmin, ymin), (xmin+w, ymin+h), (0, 255, 0), 4)
    cv2.putText(im, '{:s} {:.3f}'.format(cls, score),
                    (xmin, ymin), font, 0.5, (255, 0, 0), thickness=2)

cv2.imwrite('result.jpg', im)
plt.figure(figsize=(15,12))
plt.imshow(im[:, :, [2,1,0]])
plt.show()
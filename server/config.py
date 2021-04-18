from paddlex.det import transforms
import paddlex as pdx

from deep_sort import DeepSort

yolov3={
    'transforms':None,
    'model':None,
    'data_source':None
}

faster_rcnn={
    'transforms':transforms.Compose([
        transforms.Normalize(),
        transforms.ResizeByShort(short_size=800, max_size=1333),
        transforms.Padding(coarsest_stride=32), ]),
    'model':pdx.load_model('./models/det/FasterRCNN/best_model'),
    'data_source':pdx.datasets.VOCDetection(
    data_dir='./data/',
    file_list='data/MOT20/train/MOT20-01/manifest.txt',
    label_list='./data/labels.txt',transforms=transforms.Compose([
        transforms.Normalize(),
        transforms.ResizeByShort(short_size=800, max_size=1333),
        transforms.Padding(coarsest_stride=32), ])).file_list,
    'tracker':DeepSort('models/deep_sort/checkpoint_static/net', n_init=2)
}
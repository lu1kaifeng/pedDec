import paddlex as pdx
from paddlex.det import transforms

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32), ])
model = pdx.load_model('./FasterRCNN/best_model')
image_name = './test.jpg'
result = model.predict(image_name, eval_transforms)
pdx.det.visualize(image_name, result, threshold=0.1)

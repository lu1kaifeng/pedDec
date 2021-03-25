#from typing import Optional, Union

#import joblib
#import pytorch_lightning as pl
from pathlib2 import Path
from PIL import Image
import paddle
from paddle.io import DataLoader, Dataset
import numpy as np
#from torch.utils.data.dataset import Subset
#from torchvision import transforms

#from metrics import smooth_st_distribution
def get_ids(img_paths: list, dataset: str) -> tuple:
    camera_ids = []
    labels = []
    frames = []

    if dataset == 'market':
        dict_cam_seq_max = {
            11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346,
            17: 0, 18: 0, 21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0,
            26: 0, 27: 0, 28: 0, 31: 161708, 32: 161769, 33: 104469, 34: 0,
            35: 0, 36: 0, 37: 0, 38: 0, 41: 72107, 42: 72373, 43: 74810,
            44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0, 51: 161095,
            52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
            61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0,
            67: 0, 68: 0}

    for i, path in enumerate(img_paths):
        # File excluding the extension (.jpg)
        filename = path.stem

        if dataset == 'market':
            label, camera_seq, frame, _ = filename.split('_')
        else:
            label, camera_seq, frame = filename.split('_')
            frame = frame[1:]

        camera_id = int(camera_seq[1])
        frame = int(frame)

        if dataset == 'market':
            seq = int(camera_seq[3])
            re = 0
            for j in range(1, seq):
                re = re + dict_cam_seq_max[int(str(camera_id) + str(j))]
            frame += re

        labels.append(label)
        camera_ids.append(int(camera_id))
        frames.append(frame)
    # (list, list, list)
    return camera_ids, labels, frames



class ReIDDataset(Dataset):
    """
    The ReID Dataset module is a custom Dataset module, specific to parsing
        the Market & Duke Person Re-Identification datasets.

    Args:
        data_dir (str): The path where the dataset is located.

        transform ([list, torchvision.transforms], optional): Pass the list of
            transforms to transform the input images. Defaults to None.

        target_transform ([list, torchvision.transforms], optional): Pass the
            list of transforms to transform the labels. Defaults to None.

        ret_camid_n_frame (bool, optional): Whether to return camera ids and
            frames. True will additionally return cam_ids and frame.
            Defaults to False.

    Raises:
        Exception: If directory does not exist!

    """

    def __init__(self, data_dir: str, transform=None, target_transform=None,
                 ret_camid_n_frame: bool = False):

        super(ReIDDataset, self).__init__()
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise Exception(
                f"'Path '{self.data_dir.__str__()}' does not exist!")
        if not self.data_dir.is_dir():
            raise Exception(
                f"Path '{self.data_dir.__str__()}' is not a directory!")

        self.transform = transform
        self.target_transform = target_transform
        self.ret_camid_n_frame = ret_camid_n_frame
        self._init_data()

    def _init_data(self):

        if 'market' in str(self.data_dir).lower():
            self.dataset = 'market'
        elif 'duke' in str(self.data_dir).lower():
            self.dataset = 'duke'

        self.imgs = list(self.data_dir.glob('*.jpg'))
        # Filter out labels with -1
        self.imgs = [img for img in self.imgs if '-1' not in img.stem]

        self.cam_ids, self.labels, self.frames = get_ids(
            self.imgs, self.dataset)

        self.num_cams = len(set(self.cam_ids))
        self.classes = tuple(set(self.labels))

        # Convert labels to continuous idxs
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.targets = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        sample = Image.open(str(self.imgs[index])).convert('RGB')
        target = self.targets[index]

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        if self.ret_camid_n_frame:
            cam_id = self.cam_ids[index]
            frame = self.frames[index]
            return sample, target, cam_id, frame

        return sample , np.array([target],dtype=np.int64) #target#np.eye(len(self.classes), dtype='float32')[target]


# class ReIDDataModule(pl.LightningDataModule):
#
#     def __init__(self, data_dir: str, st_distribution: Optional[str] = None,
#                  train_subdir: str = 'bounding_box_train',
#                  test_subdir: str = 'bounding_box_test',
#                  query_subdir: str = 'query', train_batchsize: int = 16,
#                  val_batchsize: int = 16, test_batchsize: int = 16,
#                  num_workers: int = 4,
#                  random_erasing: float = 0.0, color_jitter: bool = False,
#                  save_distribution: Union[bool, str] = False):
#
#         super().__init__()
#
#         self.data_dir = Path(data_dir)
#
#         if not self.data_dir.exists():
#             raise Exception(
#                 f"'Path '{self.data_dir.__str__()}' does not exist!")
#         if not self.data_dir.is_dir():
#             raise Exception(
#                 f"Path '{self.data_dir.__str__()}' is not a directory!")
#
#         self.train_dir = self.data_dir / train_subdir
#         self.test_dir = self.data_dir / test_subdir
#         self.query_dir = self.data_dir / query_subdir
#
#         self.train_batchsize = train_batchsize
#         self.test_batchsize = test_batchsize
#         self.val_batchsize = val_batchsize
#         self.num_workers = num_workers
#
#         self.color_jitter = color_jitter
#         self.random_erasing = random_erasing
#
#         self.st_distribution = st_distribution
#         self.save_distribution = save_distribution
#
#         self.prepare_data()
#
#     def prepare_data(self):
#
#         train_transforms = [transforms.Resize((384, 192), interpolation=3),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.ToTensor(),
#                             transforms.Normalize([0.485, 0.456, 0.406], [
#                                 0.229, 0.224, 0.225])
#                             ]
#         test_transforms = train_transforms
#         test_transforms.pop(1)
#
#         if self.random_erasing > 0:
#             train_transforms.append(
#                 transforms.RandomErasing(self.random_erasing))
#         if self.color_jitter:
#             train_transforms.append(transforms.ColorJitter())
#
#         train_transforms = transforms.Compose(train_transforms)
#         self.train = ReIDDataset(self.train_dir, train_transforms)
#         self.num_classes = len(self.train.classes)
#         train_len = int(len(self.train) * 0.8)
#         test_len = len(self.train) - train_len
#         self.train, self.test = random_split(self.train, [train_len, test_len])
#
#         test_transforms = transforms.Compose(test_transforms)
#         self.test = ReIDDataset(self.test_dir, test_transforms)
#         self.query = ReIDDataset(self.query_dir, test_transforms,
#                                  ret_camid_n_frame=True)
#         self.gallery = ReIDDataset(self.test_dir, test_transforms,
#                                    ret_camid_n_frame=True)
#
#         self._load_st_distribution()
#         if self.save_distribution:
#             self._save_st_distribution()
#
#     def _load_st_distribution(self):
#
#         if isinstance(self.st_distribution, str):
#             self.st_distribution = Path(self.st_distribution)
#
#             if not (self.st_distribution.exists()
#                     and self.st_distribution.is_file()):
#                 raise FileNotFoundError(
#                     f"Location '{str(self.st_distribution)}' \
#                     does not exist or not a file!")
#
#             if self.st_distribution.suffix != '.pkl':
#                 raise ValueError('File must be of type .pkl')
#
#             print(
#                 f'\nLoading Spatial-Temporal Distribution from \
#                 {self.st_distribution}.\n\n')
#             self.st_distribution = joblib.load(str(self.st_distribution))
#
#         elif not self.st_distribution:
#             print('\n\nGenerating Spatial-Temporal Distribution.\n\n')
#             num_cams = self.query.num_cams
#             max_hist = 5000 if self.query.dataset == 'market' else 3000
#
#             cam_ids = self.query.cam_ids + self.gallery.cam_ids
#             targets = self.query.targets + self.gallery.targets
#             frames = self.query.frames + self.gallery.frames
#
#             self.st_distribution = smooth_st_distribution(cam_ids, targets,
#                                                           frames,
#                                                           num_cams, max_hist)
#
#     def _save_st_distribution(self):
#         if isinstance(self.save_distribution, str):
#             if '.pkl' not in self.save_distribution:
#                 self.save_distribution += '.pkl'
#         else:
#             self.save_distribution = self.data_dir + 'st_distribution.pkl'
#
#         print(
#             f'\nSaving distribution at {self.save_distribution}')
#         joblib.dump(self.st_distribution, self.save_distribution)
#
#     def train_dataloader(self):
#
#         return DataLoader(self.train, batch_size=self.train_batchsize,
#                           shuffle=True, num_workers=self.num_workers,
#                           pin_memory=True)
#
#     def val_dataloader(self):
#         test_loader = DataLoader(self.test, batch_size=self.test_batchsize,
#                                  shuffle=False, num_workers=self.num_workers,
#                                  pin_memory=True)
#         query_indices = range(self.test_batchsize)
#         query_loader = DataLoader(Subset(self.query, query_indices),
#                                   batch_size=self.test_batchsize,
#                                   shuffle=False, num_workers=self.num_workers,
#                                   pin_memory=True)
#         evens = list(range(0, len(self.gallery), 3))
#         gall_loader = DataLoader(Subset(self.gallery, evens),
#                                  batch_size=self.test_batchsize,
#                                  shuffle=True, num_workers=self.num_workers,
#                                  pin_memory=True)
#
#         return [query_loader, gall_loader, test_loader]
#
#     def test_dataloader(self):
#
#         query_loader = DataLoader(self.query, batch_size=self.test_batchsize,
#                                   shuffle=False, num_workers=self.num_workers,
#                                   pin_memory=True)
#         gall_loader = DataLoader(self.gallery, batch_size=self.test_batchsize,
#                                  shuffle=True, num_workers=self.num_workers,
#                                  pin_memory=True)
#
#         return [query_loader, gall_loader]

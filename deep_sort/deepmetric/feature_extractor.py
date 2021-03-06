import logging

import cv2
import numpy as np
import paddle as torch
import paddle.vision.transforms as transforms
from paddle import fluid

from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net()
        self.device = "cuda" if use_cuda else "cpu"
        state_dict = torch.load(model_path)
        self.net.set_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        # self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(data_format='HWC'),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], data_format='HWC'),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = fluid.layers.concat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], axis=0)
        im_batch = torch.cast(im_batch, 'float32')
        return im_batch

    @torch.no_grad()
    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        # im_batch = im_batch.to(self.device)
        features = self.net(im_batch)
        return features.numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

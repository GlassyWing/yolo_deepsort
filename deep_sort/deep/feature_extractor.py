import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2

from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval()
        self.size = (64, 128)

        self._mean = torch.as_tensor([0.485, 0.456, 0.406],
                                     dtype=torch.float32,
                                     device=self.device)[None, :, None, None]
        self._std = torch.as_tensor([0.229, 0.224, 0.225],
                                     dtype=torch.float32,
                                     device=self.device)[None, :, None, None]

    def __norm(self, tensor):
        tensor.sub_(self._mean).div_(self._std)
        return tensor

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
            im = torch.from_numpy(cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)).to(
                self.device).float().permute(2, 0, 1)
            return im

        im_batch = torch.stack([_resize(im, self.size) for im in im_crops], dim=0).div(255.)
        im_batch = self.__norm(im_batch)
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

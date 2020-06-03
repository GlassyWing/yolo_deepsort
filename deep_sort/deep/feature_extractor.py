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

        self.__norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        result = []
        for im in im_crops:
            im = F.interpolate(self.__norm(im).unsqueeze(0), size=(self.size[1], self.size[0]), mode="nearest")
            result.append(im)

        im_batch = torch.cat(result, dim=0)
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

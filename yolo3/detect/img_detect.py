import datetime
import logging
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patches
from matplotlib.ticker import NullLocator
from torch.autograd import Variable

from yolo3.dataset.dataset import pad_to_square, resize
from yolo3.utils.helper import load_classes
from yolo3.utils.model_build import non_max_suppression, rescale_boxes, xywh2p1p2, resize_boxes, \
    soft_non_max_suppression


def scale(image, shape, max_size):
    h, w, _ = shape
    # 按比例缩放
    if w > h:
        image = resize(image, (int(h * max_size / w), max_size))
    else:
        image = resize(image, (max_size, int(w * max_size / h)))

    if len(image.shape) != 3:
        image = image.unsqueeze(0)
        image = image.expand((3, image.shape[1:]))
    return image


class ImageDetector:
    """图像检测器，只检测单张图片"""

    def __init__(self, model, class_path, thickness=2,
                 thres=0.5,
                 nms_thres=0.4,
                 win_size=None,
                 overlap=0.15,
                 half=False):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device

        if half:
            self.model.half()

        self.classes = load_classes(class_path)
        self.num_classes = len(self.classes)
        self.thickness = thickness
        self.thres = thres
        self.nms_thres = nms_thres
        self.half = half
        self.win_size = win_size
        self.overlap = overlap

    def detect(self, img):

        h, w, _ = img.shape

        if self.win_size is not None:
            win_width, win_height = self.win_size

        if self.win_size is None or w < win_width and h < win_height:

            image = cv2.resize(img, (self.model.img_size[1], self.model.img_size[0]), interpolation=cv2.INTER_LINEAR)
            image = torch.from_numpy(image).to(self.device)
            image = image.permute((2, 0, 1)) / 255.

            # image = scale(image, img.shape, self.model.img_size)
            # image, _ = pad_to_square(image, 0)
            # image = resize(image, (self.model.img_size, self.model.img_size))

            # Add batch dimension
            image = image.unsqueeze(0)

            if self.half:
                image = image.half()

            prev_time = time.time()
            with torch.no_grad():
                detections = self.model(image)
                detections = soft_non_max_suppression(detections, self.thres, self.nms_thres)
                detections = detections[0]
                if detections is not None:
                    # detections = rescale_boxes(detections, self.model.img_size, (h, w))
                    detections = resize_boxes(detections, self.model.img_size, (h, w))

            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            logging.info("\t Inference time: %s" % inference_time)

        else:
            truncated_images = []
            truncated_images_ori_size = []

            overlap_x, overlap_y = int(win_width * self.overlap), int(win_height * self.overlap)
            offsets = []
            for x in range(0, w, win_width):
                for y in range(0, h, win_height):
                    # 截取窗口大小的图像，再加上一些重叠区域
                    img_sub = img[y:y + win_height + overlap_y, x:x + win_width + overlap_x]
                    truncated_images_ori_size.append((img_sub.shape[0], img_sub.shape[1]))

                    image = cv2.resize(img_sub, (self.model.img_size[1], self.model.img_size[0]),
                                       interpolation=cv2.INTER_LINEAR)
                    truncated_images.append(image)

                    # img = scale(img, (img.shape[1], img.shape[2], img.shape[0]), self.model.img_size)
                    # img, _ = pad_to_square(img, 0)
                    # img = resize(img, (self.model.img_size, self.model.img_size))
                    # cv2.imshow(str(x) + "-" + str(y), img.permute((1, 2, 0)).numpy())

                    offset = torch.tensor([x, y, x, y], dtype=torch.float32, device=self.device)
                    if self.half:
                        offset = offset.half()
                    offsets.append(offset)

            # (n, model.img_size, model.img_size)
            truncated_images = torch.from_numpy(np.stack(truncated_images, 0)).to(self.device)
            truncated_images = truncated_images.permute((0, 3, 1, 2)) / 255.

            if self.half:
                truncated_images = truncated_images.half()
            prev_time = time.time()
            with torch.no_grad():
                detections = self.model(truncated_images)
                detections[..., :4] = xywh2p1p2(detections[..., :4])

                rescaled_detections = []
                for idx, detection in enumerate(detections):
                    # detection = rescale_boxes(detection, self.model.img_size, truncated_images_ori_size[idx])
                    detection = resize_boxes(detection, self.model.img_size, truncated_images_ori_size[idx])
                    detection[..., :4] += offsets[idx]
                    rescaled_detections.append(detection)

                # (1, n, 5 + num_class)
                rescaled_detections = torch.cat(rescaled_detections, 0).unsqueeze(0)

                detections = soft_non_max_suppression(rescaled_detections, self.thres, self.nms_thres,
                                                      merge=True,
                                                      is_p1p2=True)
                detections = detections[0]

            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            logging.info("\t Inference time: %s" % inference_time)

        return detections


class ImageFolderDetector:
    """图像文件夹检测器，检测一个文件夹中的所有图像"""

    def __init__(self, model, class_path):
        self.model = model.eval()
        self.classes = load_classes(class_path)

    def detect(self, dataloader, output_dir, conf_thres=0.8, nms_thres=0.4):

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            input_imgs = Variable(input_imgs.type(Tensor))

            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, conf_thres, nms_thres)

            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            logging.info("\t+ Batch %d, Inference time: %s" % (batch_i, inference_time))

            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        colors = plt.get_cmap("tab20b").colors

        logging.info("\nSaving images:")

        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            logging.info("(%d) Image: '%s'" % (img_i, path))
            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            if detections is not None:
                detections = rescale_boxes(detections, self.model.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    logging.info("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=self.classes[int(cls_pred)],
                             color="white",
                             verticalalignment="top",
                             bbox={"color": color, "pad": 0})

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())

            filename = os.path.basename(path).split(".")[0]
            output_path = os.path.join(output_dir, filename + ".png")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()

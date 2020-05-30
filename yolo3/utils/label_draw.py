import logging

import cv2
import numpy as np


def _get_statistic_info(detections, unique_labels, classes):
    """获得统计信息"""
    statistic_info = {}
    for label in unique_labels:
        statistic_info[classes[int(label)]] = (detections[:, -1] == label).sum().item()
    return statistic_info


def draw_rect(img, rect, color, thickness):
    """绘制边框和标签"""
    x1, y1, x2, y2 = rect

    c1 = (int(x1), int(y1))
    c2 = (int(x2), int(y2))

    image = cv2.rectangle(img, c1, c2, color, thickness)
    return image


def draw_rect_and_label(img, rect, label, color, thickness, font, font_size=18):
    """绘制边框和标签"""
    x1, y1, x2, y2 = rect

    color = (int(color[0]), int(color[1]), int(color[2]))

    # 绘制边框
    draw_rect(img, rect, color, thickness)

    # 绘制文本框
    label_size = (int(font_size * 1.1 * len(label)), font_size)

    if y1 - label_size[1] >= 0:
        text_origin = int(x1), int(y1) - label_size[1]
    else:
        text_origin = int(x1), int(y1) + 1
    # cv2.rectangle(img, text_origin, (text_origin[0] + label_size[0],
    #                                  text_origin[1] + label_size[1]),
    #               color, -1)
    if font is not None:
        font.putText(img=img,
                     text=label,
                     org=text_origin,
                     fontHeight=font_size,
                     color=(255, 255, 255),
                     thickness=-1,
                     line_type=cv2.LINE_AA,
                     bottomLeftOrigin=False)
    else:
        cv2.putText(img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return label_size


def draw_single_img(img, detections, img_size,
                    classes,
                    colors,
                    thickness, font,
                    statistic=False,
                    scaled=False,
                    only_rect=False,
                    font_size=18):
    """绘制单张图片"""
    statistic_info = {}

    # Detected something
    plane = np.zeros_like(img)
    if detections is not None:

        if statistic:
            unique_labels = detections[:, -1].unique()
            statistic_info = _get_statistic_info(detections, unique_labels, classes)

        # make a blank image for text, rectangle, initialized to transparent color

        font_height = 0
        font_width = 0

        for idx, detection in enumerate(detections):
            if only_rect:
                draw_rect(plane, detection[:4], colors[int(detection[-1])], thickness)

            else:
                # 绘制所有标签
                (fw, fh) = draw_rect_and_label(plane, detection[:4],
                                               classes[int(detection[-1])],
                                               colors[int(detection[-1])],
                                               thickness,
                                               font,
                                               font_size)
                font_height = max(font_height, fh)
                font_width = max(font_width, fw)

        if not only_rect and statistic:
            # 绘制统计信息
            pass

        if len(detections) > 0:
            plane2gray = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)

            # 将像素值大于0的全都设为黑色，为0的全都为白色
            _, mask_inv = cv2.threshold(plane2gray, 0, 255, cv2.THRESH_BINARY_INV)
        else:
            mask_inv = None

        return img, plane, mask_inv

    else:
        logging.debug("Nothing Detected.")
        return img, None, None


def plane_composite(img, plane, plane_mask):
    """将背景层与绘制检测框的图层叠加"""

    # 在原图中，把要添加的部分设置为黑色
    img_bg = cv2.bitwise_and(img, img, mask=plane_mask)

    return cv2.add(img_bg, plane)


class LabelDrawer:
    """绘制便签工具"""

    def __init__(self,
                 classes,
                 font_path,
                 font_size,
                 thickness,
                 img_size,
                 statistic=False,
                 id2label=None):
        self.thickness = thickness
        self.statistic = statistic
        self.classes = classes
        self.img_size = img_size
        self.font_size = font_size
        self.id2label = id2label
        self.font_path = font_path

        num_classes = len(self.classes)
        if font_path is not None:
            self.font = cv2.freetype.createFreeType2()
            self.font.loadFontData(fontFileName=font_path, id=0)
        else:
            self.font = None

        # Prepare colors for each class
        np.random.seed(1)
        self.colors = (np.random.rand(min(999, num_classes), 3) * 255).astype(int)
        np.random.seed(None)

    def clone(self):
        return LabelDrawer(self.classes, self.font_path, self.font_size, self.thickness,
                           self.img_size, self.statistic, None)

    def draw_labels(self, img, detections, only_rect, scaled=True):
        return draw_single_img(img, detections, self.img_size, self.classes,
                               self.colors,
                               self.thickness,
                               self.font,
                               statistic=self.statistic,
                               scaled=scaled,
                               only_rect=only_rect,
                               font_size=self.font_size)

    def draw_labels_by_trackers(self, img, detections, only_rect):
        statistic_info = {}

        plane = np.zeros_like(img)
        for detection in detections:

            font_height = 0
            font_width = 0

            if only_rect:
                draw_rect(plane, detection[:4], self.colors[int(detection[4])], self.thickness)

            else:

                if self.id2label is not None and str(int(detection[4])) in self.id2label:
                    label = str(int(detection[4])) + ":" + self.id2label[str(int(detection[4]))]
                else:
                    label = str(int(detection[4])) + ":" + self.classes[int(detection[-1])]

                # 绘制所有标签
                fw, fh = draw_rect_and_label(plane,
                                             detection[:4],
                                             label,
                                             self.colors[int(detection[-1]) % len(self.colors)],
                                             self.thickness,
                                             self.font,
                                             font_size=self.font_size)
                font_height = max(font_height, fh)
                font_width = max(font_width, fw)

            if self.statistic:
                pass

        if len(detections) > 0:
            plane2gray = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)

            # 将像素值大于0的全都设为黑色，为0的全都为白色
            _, mask_inv = cv2.threshold(plane2gray, 0, 255, cv2.THRESH_BINARY_INV)
        else:
            mask_inv = None

        return img, plane, mask_inv

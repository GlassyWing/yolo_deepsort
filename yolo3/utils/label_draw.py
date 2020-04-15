import colorsys
import logging
from concurrent.futures.thread import ThreadPoolExecutor
import random
import cv2

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolo3.utils.model_build import rescale_boxes


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


def draw_summary(draw, font, summary, font_width, font_height, y_offset=60):
    """绘制摘要信息"""
    draw.rectangle(xy=((0, y_offset), (font_width, font_height * len(summary) + y_offset)),
                   fill=(0, 0, 0, 128))

    with ThreadPoolExecutor() as executor:
        for _ in executor.map(
                lambda x: draw.text((3, x[0] * font_height + y_offset), x[1][0] + ":" + str(x[1][1]),
                                    fill=(255, 255, 255, 128),
                                    font=font), enumerate(summary.items())):
            pass


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

    plane = np.zeros(img.shape, np.uint8)

    # Detected something
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
            # draw_summary(draw, font=font,
            #              summary=statistic_info,
            #              font_width=font_width,
            #              font_height=font_height
            #              )
            pass

        return img, plane, statistic_info

    else:
        logging.debug("Nothing Detected.")
        return img, None, statistic_info


def plane_composite(img, plane):
    """将背景层与绘制检测框的图层叠加"""

    plane2gray = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)

    # 将像素值大于0的全都设为白色，为0的全都为黑色
    _, mask = cv2.threshold(plane2gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 在原图中，把要添加的部分设置为黑色
    img_bg = cv2.bitwise_and(img, img, mask=mask_inv)

    # 在前景中，把不添加的部分设置为黑色
    plane_fg = cv2.bitwise_and(plane, plane, mask=mask)

    return cv2.add(img_bg, plane_fg)


class LabelDrawer:
    """绘制便签工具"""

    def __init__(self,
                 classes,
                 font_path,
                 font_size,
                 thickness,
                 img_size,
                 statistic=False):
        self.thickness = thickness
        self.statistic = statistic
        self.classes = classes
        self.img_size = img_size
        self.font_size = font_size

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

    def draw_labels(self, img, detections, only_rect, scaled=True):
        return draw_single_img(img, detections, self.img_size, self.classes,
                               self.colors,
                               self.thickness,
                               self.font,
                               statistic=self.statistic,
                               scaled=scaled,
                               only_rect=only_rect,
                               font_size= self.font_size)

    def draw_labels_by_trackers(self, img, detections, only_rect):
        statistic_info = {}

        # make a blank image for text, rectangle, initialized to transparent color
        plane = np.zeros(img.shape, np.uint8)

        for detection in detections:

            font_height = 0
            font_width = 0

            if only_rect:
                draw_rect(plane, detection[:4], self.colors[int(detection[-1])], self.thickness)

            else:
                # 绘制所有标签
                fw, fh = draw_rect_and_label(plane,
                                             detection[:4],
                                             # str(tracker.track_id) + ":" + self.classes[int(classId)],
                                             str(int(detection[-1])),
                                             self.colors[int(detection[-1]) % len(self.colors)],
                                             self.thickness,
                                             self.font,
                                             font_size=self.font_size)
                font_height = max(font_height, fh)
                font_width = max(font_width, fw)

            if self.statistic:
                # 绘制统计信息
                # draw_summary(draw, font=self.font,
                #              summary=statistic_info,
                #              font_width=font_width,
                #              font_height=font_height
                #              )
                pass

        return img, plane, statistic_info

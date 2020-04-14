import colorsys
import logging
from concurrent.futures.thread import ThreadPoolExecutor
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolo3.utils.model_build import rescale_boxes


def _get_statistic_info(detections, unique_labels, classes):
    """获得统计信息"""
    statistic_info = {}
    for label in unique_labels:
        statistic_info[classes[int(label)]] = (detections[:, -1] == label).sum().item()
    return statistic_info


def draw_rect(draw, rect, color, thickness):
    """绘制边框和标签"""
    x1, y1, x2, y2 = rect

    c1 = (int(x1), int(y1))
    c2 = (int(x2), int(y2))

    draw.rectangle([c1, c2], outline=tuple(color) + (128,), width=thickness)


def draw_rect_and_label(draw, rect, label, color, thickness, font):
    """绘制边框和标签"""
    x1, y1, x2, y2 = rect

    c1 = (int(x1), int(y1))
    c2 = (int(x2), int(y2))

    draw.rectangle([c1, c2], outline=tuple(color) + (128,), width=thickness)

    # 绘制文本框
    label_size = draw.textsize(label, font)

    if y1 - label_size[1] >= 0:
        text_origin = int(x1), int(y1) - label_size[1]
    else:
        text_origin = int(x1), int(y1) + 1
    draw.rectangle([text_origin, (text_origin[0] + label_size[0],
                                  text_origin[1] + label_size[1])],
                   fill=tuple(color) + (128,))
    draw.text(xy=text_origin, text=label, fill=(255, 255, 255), font=font)

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
                    only_rect=False):
    """绘制单张图片"""
    statistic_info = {}

    # Detected something
    if detections is not None:

        # 使用PIL绘制中文
        base = Image.fromarray(img).convert("RGBA")
        w, h = base.size

        # 如果检测框尚未进行缩放
        if not scaled:
            detections = rescale_boxes(detections, img_size, (h, w))

        if statistic:
            unique_labels = detections[:, -1].unique()
            statistic_info = _get_statistic_info(detections, unique_labels, classes)

        # make a blank image for text, rectangle, initialized to transparent color
        plane = Image.new("RGBA", base.size, (255, 255, 255, 0))

        draw = ImageDraw.Draw(plane)

        font_height = 0
        font_width = 0

        for idx, detection in enumerate(detections):
            if only_rect:
                draw_rect(draw, detection[:4], colors[int(detection[-1])], thickness)

            else:
                # 绘制所有标签
                fw, fh = draw_rect_and_label(draw, detection[:4],
                                             classes[int(detection[-1])],
                                             colors[int(detection[-1])],
                                             thickness,
                                             font)
                font_height = max(font_height, fh)
                font_width = max(font_width, fw)

        if not only_rect and statistic:
            # 绘制统计信息
            draw_summary(draw, font=font,
                         summary=statistic_info,
                         font_width=font_width,
                         font_height=font_height
                         )
        del draw

        out = Image.alpha_composite(base, plane).convert("RGB")
        img = np.ndarray(buffer=out.tobytes(), shape=img.shape, dtype='uint8', order='C')

        return img, plane, statistic_info

    else:
        logging.debug("Nothing Detected.")
        return img, None, statistic_info


class LabelDrawer:

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

        num_classes = len(self.classes)
        if font_path is not None:
            self.font = ImageFont.truetype(font_path, font_size)
        else:
            self.font = ImageFont.load_default()

        # Prepare colors for each class
        hsv_color = [(1.0 * i / num_classes, 1., 1.) for i in range(num_classes)]
        colors = [colorsys.hsv_to_rgb(*x) for x in hsv_color]
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        self.colors = (np.random.rand(num_classes, 3) * 255).astype(int)

    def draw_labels(self, img, detections, only_rect, scaled=True):
        return draw_single_img(img, detections, self.img_size, self.classes,
                               self.colors,
                               self.thickness,
                               self.font,
                               statistic=self.statistic,
                               scaled=scaled,
                               only_rect=only_rect)

    def draw_labels_by_trackers(self, img, detections, only_rect):
        statistic_info = {}

        # 使用PIL绘制中文
        base = Image.fromarray(img).convert("RGBA")
        # make a blank image for text, rectangle, initialized to transparent color
        plane = Image.new("RGBA", base.size, (255, 255, 255, 0))

        draw = ImageDraw.Draw(plane)

        for detection in detections:

            font_height = 0
            font_width = 0

            if only_rect:
                draw_rect(draw, detection[:4], self.colors[int(detection[-1])], self.thickness)

            else:
                # 绘制所有标签
                fw, fh = draw_rect_and_label(draw,
                                             detection[:4],
                                             # str(tracker.track_id) + ":" + self.classes[int(classId)],
                                             str(int(detection[-1])),
                                             self.colors[int(detection[-1]) % len(self.colors)],
                                             self.thickness,
                                             self.font)
                font_height = max(font_height, fh)
                font_width = max(font_width, fw)

            if self.statistic:
                # 绘制统计信息
                draw_summary(draw, font=self.font,
                             summary=statistic_info,
                             font_width=font_width,
                             font_height=font_height
                             )
        del draw

        out = Image.alpha_composite(base, plane).convert("RGB")
        img = np.ndarray(buffer=out.tobytes(), shape=img.shape, dtype='uint8', order='C')

        return img, plane, statistic_info

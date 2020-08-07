import logging

import cv2
import numpy as np
import time


def _get_statistic_info(detections, unique_labels, classes):
    """获得统计信息"""
    statistic_info = {}
    for label in unique_labels:
        statistic_info[classes[int(label)]] = (
                detections[:, -1] == label).sum().item()
    return statistic_info


def draw_rects(img, dets, colors, thickness):
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        cls = int(det[-1])

        c1 = (int(x1), int(y1))
        c2 = (int(x2), int(y2))

        cv2.rectangle(img, c1, c2, colors[cls], thickness)
    return img


def draw_rects_and_labels(img, dets, colors, labels, thickness, font_size, font=None):
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = det[:4]
        cls = int(det[-1])

        c1 = (int(x1), int(y1))
        c2 = (int(x2), int(y2))

        cv2.rectangle(img, c1, c2, colors[cls], thickness)

        if font is not None:
            text_size, _ = font.getTextSize(labels[i], font_size, -1)
            font_w, font_h = text_size
            cv2.rectangle(img, (c1[0], max(0, c1[1] - 3 - font_size)),
                          (c1[0] + font_w, max(c1[1], 3 + font_size)), colors[cls], -1)
            font.putText(img=img,
                         text=labels[i],
                         org=(c1[0], max(c1[1] - 3, font_size)),
                         fontHeight=font_size,
                         color=(0, 0, 0),
                         thickness=-1,
                         line_type=cv2.LINE_4,
                         bottomLeftOrigin=True)
        else:
            text_size, _ = cv2.getTextSize(
                labels[i], cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, 1)
            font_w, font_h = text_size
            cv2.rectangle(img, (c1[0], max(0, int(c1[1] - 3 - 18 * font_size))),
                          (c1[0] + font_w, max(c1[1], int(3 + 18 * font_size))), colors[cls], -1)
            cv2.putText(img,
                        labels[i],
                        (c1[0], max(c1[1] - 3, font_h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size,
                        (0, 0, 0), 1)
    return img


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
    if detections is not None:
        detections = detections.cpu().float().numpy()
        if statistic:
            unique_labels = detections[:, -1].unique()
            statistic_info = _get_statistic_info(
                detections, unique_labels, classes)

        if only_rect:
            draw_rects(img, detections, colors, thickness)
        else:
            labels = []
            for detection in detections:
                if len(detection) == 7:
                    labels.append(classes[int(detection[-1])] +
                                  ' (' + str(round(detection[-3] * detection[-2] * 100, 2)) + '%)')
                else:
                    labels.append(classes[int(detection[-1])] +
                                  ' (' + str(round(detection[-2] * 100, 2)) + '%)')
            draw_rects_and_labels(img, detections, colors,
                                  labels, thickness, font_size, font)

        if not only_rect and statistic:
            # 绘制统计信息
            pass

        return img, None, None

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
        self.colors = (np.random.rand(
            min(999, num_classes), 3) * 255).astype(int)
        np.random.seed(None)
        self.colors = [(int(color[0]), int(color[1]), int(color[2]))
                       for color in self.colors]

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
                               font_size=img.shape[0] / 1000. if self.font is None else self.font_size)

    def draw_labels_by_trackers(self, img, detections, only_rect):

        if only_rect:
            draw_rects(img, detections, self.colors, self.thickness)
        else:

            labels = []
            for detection in detections:
                if self.id2label is not None and str(int(detection[4])) in self.id2label:
                    label = str(int(detection[4])) + ":" + \
                            self.id2label[str(int(detection[4]))]
                else:
                    label = str(int(detection[4])) + ":" + \
                            self.classes[int(detection[-1])]
                labels.append(label)

            # 绘制所有标签
            draw_rects_and_labels(img,
                                  detections,
                                  self.colors,
                                  labels,
                                  self.thickness,
                                  img.shape[0] / 1000. if self.font is None else self.font_size,
                                  self.font)

        return img, None, None

import logging
import time
from functools import reduce

import cv2
import numpy as np
from PIL import Image

from yolo3.detect.img_detect import ImageDetector
from yolo3.utils.helper import load_classes
from yolo3.utils.label_draw import LabelDrawer, plane_composite
from yolo3.utils.model_build import p1p2Toxywh


def alpha_composite(img, plane):
    if plane is None:
        return img
    else:

        base = Image.fromarray(img)
        base = base.convert("RGBA")
        out = Image.alpha_composite(base, plane).convert("RGB")
        result = np.ndarray(buffer=out.tobytes(), shape=img.shape, dtype='uint8')

        return result


class VideoDetector:
    """视频检测器，用于检测视频"""

    def __init__(self, model, class_path,
                 thickness=2,
                 font_path=None,
                 font_size=10,
                 conf_thres=0.7,
                 nms_thres=0.4,
                 skip_frames=-1,
                 fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                 class_mask=None,
                 win_size=None,
                 overlap=0.15,
                 tracker=None,
                 action_id=None):
        self.thickness = thickness
        self.skip_frames = skip_frames
        self.class_mask = class_mask
        self.fourcc = fourcc

        self.label_drawer = LabelDrawer(load_classes(class_path),
                                        font_path,
                                        font_size,
                                        thickness,
                                        img_size=model.img_size)

        self.image_detector = ImageDetector(model, class_path,
                                            thickness=thickness,
                                            conf_thres=conf_thres,
                                            nms_thres=nms_thres,
                                            win_size=win_size,
                                            overlap=overlap)

        self.tracker = tracker
        self.action_id = action_id

    def detect(self, video_path,
               output_path=None,
               skip_secs=0,
               real_show=False,
               show_fps=True,
               ):
        logging.info("Detect video: " + str(video_path))
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = int(vid.get(cv2.CAP_PROP_FPS))
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = int(skip_secs) * video_fps
        if skip_secs > total_frames:
            print("Can't skip over total video!")
        else:
            vid.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

        isOutput = True if output_path is not None else False
        if isOutput:
            logging.info(f"Output Type: {output_path}, {video_FourCC}, {video_fps}, {video_size}")
            out = cv2.VideoWriter(output_path, self.fourcc, video_fps, video_size)

        if real_show:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("result", 800, 600)

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = time.time()

        hold_plane_mask = None
        hold_plane = None
        hold_detections = None
        actions = []

        frames = 0
        try:
            while True:

                return_value, frame = vid.read()
                if not return_value:
                    break

                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frames % self.skip_frames == 0:
                    detections = self.image_detector.detect(frame)

                    if detections is not None and self.tracker is not None:
                        boxs = p1p2Toxywh(detections[:, :4])
                        class_ids = detections[:, -1]
                        confidences = detections[:, 4]
                        if self.class_mask is not None:
                            mask_set = [class_ids == mask_id for mask_id in self.class_mask]
                            mask = reduce(lambda a, b: a | b, mask_set)

                            boxs = boxs[mask]
                            confidences = confidences[mask]
                            class_ids = class_ids[mask]

                        detections = self.tracker.update(boxs.cpu(), confidences, frame, class_ids)

                        image, plane, plane_mask = self.label_drawer.draw_labels_by_trackers(frame,
                                                                                             detections,
                                                                                             only_rect=False)

                        if self.action_id is not None:
                            actions = self.action_id.update(detections)
                        else:
                            actions = []
                    else:
                        image, plane, plane_mask = self.label_drawer.draw_labels(frame, detections,
                                                                                 only_rect=False)

                    hold_plane_mask = plane_mask
                    hold_plane = plane
                    hold_detections = detections
                    frames = 0
                else:
                    actions = []

                if hold_plane is None:
                    image = frame
                else:
                    image = plane_composite(frame, hold_plane, hold_plane_mask)

                # RGB -> BGR
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                frames += 1

                curr_time = time.time()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time += exec_time
                curr_fps += 1

                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0

                if show_fps:
                    cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(255, 0, 0), thickness=self.thickness)

                # Show the video in real time.
                if real_show:
                    cv2.imshow("result", result)

                if isOutput:
                    out.write(result)
                yield result, hold_detections, actions

                if real_show:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            if isOutput:
                out.release()
            if real_show:
                cv2.destroyAllWindows()

import datetime
import logging
import time

from action.action_Identify import ActionIdentify
from action.actions import TakeOff, Landing, Glide
from deep_sort import DeepSort
from yolo3.detect.video_detect import VideoDetector
from yolo3.models import Darknet

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    model = Darknet("config/yolov3.cfg", img_size=512)
    model.load_darknet_weights("weights/yolov3.weights")
    model.to("cuda:0")

    tracker = DeepSort("weights/ckpt.t7", min_confidence=0.5, nms_max_overlap=1)

    video_detector = VideoDetector(model, "config/coco.names",
                                   font_path="font/sarasa-bold.ttc",
                                   font_size=12,
                                   skip_frames=2,
                                   conf_thres=0.5,
                                   nms_thres=0.2,
                                   tracker=tracker)

    action_id = ActionIdentify(actions=[TakeOff(delta=4), Landing(delta=0), Glide(delta=4)], max_size=8)

    frames = 0
    for image, detections in video_detector.detect("E:/projects/python/data/landing.flv",
                                                   # output_path="../data/output.ts",
                                                   real_show=True,
                                                   show_statistic=True,
                                                   skip_times=55):

        # 检测动作
        action_id.update(detections)

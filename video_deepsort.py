import logging

from action.action_Identify import ActionIdentify
from action.actions import *
from deep_sort import DeepSort
from deep_sort.deep.config import setup_cfg
from yolo3.detect.video_detect import VideoDetector
from yolo3.models import Darknet

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    device = "cpu"

    # Initialize Darknet.
    model = Darknet("config/yolo-obj-3l.cfg", img_size=(608, 608))
    model.load_darknet_weights("weights/yolo-obj_best.weights")
    model.to(device)

    # Initialize FastReid and DeepSort

    # Set params for fast-reid
    cfg = setup_cfg({"config_file": "config/deep/config_darknet.yaml",
                     "opts": ["MODEL.WEIGHTS",
                              "weights/model_darknet.pth",
                              "MODEL.DEVICE",
                              device]})

    # 跟踪器
    tracker = DeepSort(cfg,
                       min_confidence=1,
                       nn_budget=30,
                       n_init=3,
                       max_iou_distance=0.7,
                       max_dist=0.5,
                       max_age=30)

    # Action Identify
    # action_id = ActionIdentify(actions=[TakeOff(4, delta=(0, 1)),
    #                                     Landing(4, delta=(2, 2)),
    #                                     Glide(4, delta=(1, 2)),
    #                                     FastCrossing(4, speed=0.2),
    #                                     BreakInto(0, timeout=2)],
    #                            max_age=30,
    #                            max_size=8)

    video_detector = VideoDetector(model, "config/obj.names",
                                   #font_path="font/NotoSansSC-Regular.otf",
                                   #font_size=14,
                                   thickness=2,
                                   skip_frames=1,
                                   thres=0.18,
                                   class_mask=[0, 2, 4],
                                   nms_thres=0.4,
                                   tracker=tracker,
                                   half=False)

    for image, detections, _ in video_detector.detect(
            "E:/projects/python/data/TownCentreXVID.avi",
            # output_path="../data/output.ts",
            real_show=True,
            skip_secs=0):
        # print(detections)
        pass

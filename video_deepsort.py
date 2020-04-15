import logging

from deep_sort import DeepSort
from yolo3.detect.video_detect import VideoDetector
from yolo3.models import Darknet

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    model = Darknet("config/yolov3.cfg", img_size=512)
    model.load_darknet_weights("weights/yolov3.weights")
    model.to("cuda:0")

    tracker = DeepSort("weights/ckpt.t7", min_confidence=0.5, nms_max_overlap=0.4)
    # tracker = Sort(max_age=30)

    video_detector = VideoDetector(model, "config/coco.names",
                                   font_path="font/sarasa-bold.ttc",
                                   font_size=12,
                                   skip_frames=2,
                                   conf_thres=0.5,
                                   nms_thres=0.4,
                                   tracker=tracker)

    frames = 0
    for image in video_detector.detect("E:/python/data/toky.mp4",
                                       # output_path="../data/output.ts",
                                       real_show=True,
                                       show_statistic=True,
                                       skip_times=30):
        # if frames > 10:
        #     break
        # frames += 1
        pass

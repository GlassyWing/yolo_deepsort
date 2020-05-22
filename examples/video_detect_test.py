import logging

from yolo3.detect.video_detect import VideoDetector
from yolo3.models import Darknet
from yolo3.track.deep_sort import DeepSort
from keras import backend

backend.clear_session()

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    model = Darknet("../config/yolov3.cfg", img_size=512)
    model.load_darknet_weights("../weights/yolov3.weights")
    model.to("cuda:0")

    tracker = DeepSort("model_data/market1501.pb", 0.5, 0.2)

    video_detector = VideoDetector(model, "../config/coco.names",
                                   font_path="../font/sarasa-bold.ttc",
                                   font_size=18,
                                   skip_frames=-1,
                                   conf_thres=0.7,
                                   nms_thres=0.2,
                                   tracker=tracker)

    frames = 0
    for image in video_detector.detect("../data/f35.flv",
                                       # output_path="../data/output.ts",
                                       real_show=True,
                                       show_statistic=True,
                                       skip_secs=0):
        # if frames > 10:
        #     break
        # frames += 1
        pass

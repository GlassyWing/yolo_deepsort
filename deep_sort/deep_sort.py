import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.tracker import Tracker

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100,
                 use_cuda=False):
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.use_cuda = use_cuda

        if type(model_path) == str:
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
        else:
            self.extractor = model_path

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric,
                               max_iou_distance=max_iou_distance,
                               max_age=max_age,
                               n_init=n_init,
                               use_cuda=use_cuda)

    def clone(self):
        return DeepSort(self.extractor, self.max_dist, self.min_confidence, self.nms_max_overlap, self.max_iou_distance,
                        self.max_age, self.n_init, self.nn_budget,
                        self.use_cuda)

    def update(self, bbox_xywh, confidences, ori_img, payload):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = bbox_xywh.to(self.tracker.device)
        detections = [Detection(bbox_tlwh[i], conf, features[i], payload[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        if self.nms_max_overlap != 1:
            # run on non-maximum supression
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        boxes = []
        valid_tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            boxes.append(track.mean)
            valid_tracks.append(track)

        if len(valid_tracks) != 0:
            # (n, 4)
            boxes = torch.cat(boxes, dim=0)[:, :4]
            boxes[:, 2] *= boxes[:, 3]
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes = self._tlwh_to_xyxy(boxes)

        for idx, track in enumerate(valid_tracks):
            x1, y1, x2, y2 = boxes[idx]
            track_id = track.track_id
            class_id = track.payload
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        # bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        # bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        """
        bbox_tlwh[:, 2:] += bbox_tlwh[:, :2]
        bbox_tlwh[:, :2] = torch.clamp(bbox_tlwh[:, :2], min=0)

        return bbox_tlwh

    def _s_tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w * 0.75), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h * 0.75), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []

        ori_img = torch.from_numpy(ori_img.astype(np.uint8)).to(self.extractor.device)
        ori_img = ori_img.permute((2, 0, 1)) / 255.

        for box in bbox_xywh:
            x1, y1, x2, y2 = self._s_tlwh_to_xyxy(box)
            im = ori_img[:, y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops).to(self.tracker.device)
        else:
            features = np.array([])
        return features

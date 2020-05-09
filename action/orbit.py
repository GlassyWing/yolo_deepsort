from collections import deque


class Orbit:

    def __init__(self, max_age, track_id, class_id):
        self.max_age = max_age
        self.track_id = track_id
        self.class_id = class_id
        self.deque = deque(maxlen=max_age)
        self.age = 0

    @staticmethod
    def _center_point(bbox):
        """获得中心点"""
        center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
        center_y = bbox[3]
        return center_x, center_y

    def update(self, detection):
        self.age = 0
        self.deque.append(Orbit._center_point(detection[:4]))


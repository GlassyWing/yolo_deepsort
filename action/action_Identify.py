from action.orbit import Orbit


class ActionIdentify:

    def __init__(self, actions, max_age=30, max_size=4):
        self.cache = {}
        self.max_age = max_age
        self.max_size = max_size
        self.actions = actions

    def update(self, detections):
        if detections is None:
            return

        targets = []
        for detection in detections:
            targets.append(detection[4])
            if detection[4] not in self.cache:
                self.cache[detection[4]] = Orbit(self.max_size, detection[4], detection[-1])
            else:
                self.cache[detection[4]].update(detection)

        to_be_del = []

        # 删除超时的轨迹
        for track_id, orbit in self.cache.items():
            if track_id not in targets:
                orbit.age += 1
                if orbit.age >= self.max_age:
                    to_be_del.append(track_id)

        for track_id in to_be_del:
            del self.cache[track_id]

        # 识别动作
        for track_id, orbit in self.cache.items():
            if orbit.age < self.max_size:
                for action in self.actions:
                    if action.confirm(orbit):
                        return track_id, action.name
                        break

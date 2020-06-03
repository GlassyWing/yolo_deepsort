from abc import abstractmethod, ABC

from action.orbit import Orbit


class Action:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def confirm(self, orbit):
        pass

    @staticmethod
    def action_detect(orbit):
        for action_cls in Action.__subclasses__():
            act = action_cls()
            if act.confirm(orbit):
                return act
        return None


class TakeOff(Action, ABC):
    """起飞动作，简单根据y轴上的变化判断"""

    def __init__(self, class_id, delta):
        self.delta = delta
        self.class_id = class_id
        super().__init__("takeoff")

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != self.class_id:
            return False

        is_takeoff = False
        ori_center = orbit.deque[0]
        for idx in range(1, len(orbit.deque)):
            center = orbit.deque[idx]
            if ori_center[1] - center[1] > self.delta[1] and abs(ori_center[0] - center[0]) > self.delta[0]:
                if idx == 1:
                    is_takeoff = True
                else:
                    is_takeoff = is_takeoff & True
            else:
                is_takeoff = False
            ori_center = center

        return is_takeoff


class Landing(Action, ABC):
    """降落动作，简单根据y轴方向上的变化判断"""

    def __init__(self, class_id, delta):
        self.delta = delta
        self.class_id = class_id
        super().__init__("landing")

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != self.class_id:
            return False

        if_landing = False
        ori_center = orbit.deque[0]
        for idx in range(1, len(orbit.deque)):
            center = orbit.deque[idx]
            if center[1] - ori_center[1] > self.delta[1] and abs(ori_center[0] - center[0]) > self.delta[0]:
                if idx == 1:
                    if_landing = True
                else:
                    if_landing = if_landing & True
            else:
                if_landing = False
            ori_center = center

        return if_landing


class Glide(Action, ABC):
    """滑行动作，当飞机在y轴上不怎么变化，x轴上移动时认为是滑行"""

    def __init__(self, class_id, delta):
        self.delta = delta
        self.class_id = class_id
        super().__init__("glide")

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != self.class_id:
            return False
        is_glide = False
        ori_center = orbit.deque[0]
        for idx in range(1, len(orbit.deque)):
            center = orbit.deque[idx]
            if abs(center[1] - ori_center[1]) < self.delta[1] and abs(center[0] - ori_center[0]) > self.delta[0]:
                if idx == 1:
                    is_glide = True
                else:
                    is_glide = is_glide & True
            else:
                is_glide = False
            ori_center = center

        return is_glide


class FastCrossing(Action, ABC):
    """快速穿越"""

    def __init__(self, class_id, speed):
        super().__init__("fast_crossing")
        self.class_id = class_id
        self.speed = speed

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != self.class_id:
            return False
        is_confirm = False
        ori_center = orbit.deque[0]
        ori_timestamp = orbit.timestamps[0]
        for idx in range(1, len(orbit.deque)):
            center = orbit.deque[idx]
            timestamp = orbit.timestamps[idx]
            if abs(center[0] - ori_center[0]) / ((timestamp - ori_timestamp) * 1000) > self.speed:
                if idx == 1:
                    is_confirm = True
                else:
                    is_confirm = is_confirm & True
            else:
                is_confirm = False
            ori_center = center
            ori_timestamp = timestamp
        return is_confirm


class BreakInto(Action, ABC):
    """闯入"""

    def __init__(self, class_id, timeout):
        super().__init__("break_into")
        self.class_id = class_id
        self.timeout = timeout

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != self.class_id:
            return False
        is_confirm = False
        if len(orbit.deque) > self.timeout:
            is_confirm = True
        return is_confirm

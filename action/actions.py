from abc import abstractmethod, ABC

from action.orbit import Orbit
import numpy as np


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

    def __init__(self, delta):
        self.delta = delta
        super().__init__("takeoff")

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != 4:
            return False

        is_takeoff = True
        ori_center = orbit.deque[0]
        for idx in range(1, len(orbit.deque)):
            center = orbit.deque[idx]
            if ori_center[1] - center[1] > self.delta:
                is_takeoff = is_takeoff & True
            else:
                is_takeoff = False
            ori_center = center

        return is_takeoff


class Landing(Action, ABC):
    """降落动作，简单根据y轴方向上的变化判断"""

    def __init__(self, delta):
        self.delta = delta
        super().__init__("landing")

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != 4:
            return False

        if_landing = True
        ori_center = orbit.deque[0]
        for idx in range(1, len(orbit.deque)):
            center = orbit.deque[idx]
            if center[1] - ori_center[1] > self.delta:
                if_landing = if_landing & True
            else:
                if_landing = False
            ori_center = center

        return if_landing


class Glide(Action, ABC):
    """滑行动作，当飞机在y轴上不怎么变化，x轴上移动时认为是滑行"""

    def __init__(self, delta):
        self.delta = delta
        super().__init__("glide")

    def confirm(self, orbit: Orbit):
        if len(orbit.deque) == 0 or orbit.class_id != 4:
            return False
        is_clide = False
        ori_center = orbit.deque[0]
        for idx in range(1, len(orbit.deque)):
            center = orbit.deque[idx]
            if abs(center[1] - ori_center[1]) < self.delta and abs(center[0] - ori_center[0]) > 0:
                is_clide = True
            else:
                is_clide = False
            ori_center = center

        return is_clide

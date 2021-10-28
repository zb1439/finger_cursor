import sys

from finger_cursor.driver import MacMouse, WinMouse
from finger_cursor.utils import queue, Registry


CONTROLLER = Registry("CONTROLLER")


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history = cfg.CONTROLLER.HISTORY
        self.scale = cfg.CONTROLLER.SCALE
        self.cls_queue = queue(cfg.MODEL.CLASSIFIER.NAME)
        self.det_queue = queue(cfg.MODEL.DETECTOR.NAME)
        if sys.platform == "darwin":
            self.mouse = MacMouse()
        elif sys.platform == "win32":
            self.mouse = WinMouse()
        else:
            raise NotImplementedError("Unsupported os " + sys.platform)

    def apply(self, gestures, coords):
        """
        :param gestures: output from the classifier queue
        :param coords: output from the detector queue
        :return: coordinates to move, and mouse events will be raise in this function
        """
        raise NotImplementedError

    def __call__(self):
        gestures = self.cls_queue[-self.history:]
        coords = self.det_queue[-self.history:]
        coords = [[c[0] * self.scale[0], c[1] * self.scale[1]] for c in coords]
        coord = self.apply(gestures, coords)
        return gestures[-1], coord


@CONTROLLER.register()
class SimpleRelativeController(Controller):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pressing = False
        self.right_key = False

    def apply(self, gestures, coords):
        gesture = gestures[-1]
        if len(coords) >= 2:
            dx = coords[-1][0] - coords[-2][0]
            dy = coords[-1][1] - coords[-2][1]
        else:
            dx = 0
            dy = 0

        if gesture == "drag":
            self.mouse.press()
            self.pressing = True
            self.mouse.move_rel(dx, dy)
        elif gesture == "click":
            self.mouse.press(right=False)
            self.mouse.release(right=False)
            return 0, 0
        elif gesture == "right-click":
            self.mouse.press(right=True)
            self.mouse.release(right=True)
            return 0, 0
        elif self.pressing:
            self.mouse.release(right=self.right_key)
            self.right_key = False
            self.pressing = False

        if gesture == "swipe" and len(gestures) >= 2 and gestures[-2] == "swipe":
            dy /= 10
            if dy > 0:
                self.mouse.scroll_down(dy)
            else:
                self.mouse.scroll_up(-dy)
            return 0, dy
        elif gesture == "point" and len(gestures) >= 2 and gestures[-2] == "point":
            self.mouse.move_rel(dx, dy)
        return dx, dy


@CONTROLLER.register()
class SimpleAbsoluteController(Controller):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pressing = False
        self.right_key = False

    def apply(self, gestures, coords):
        gesture = gestures[-1]
        coord = coords[-1]
        if len(coords) >= 2:
            dx = coords[-1][0] - coords[-2][0]
            dy = coords[-1][1] - coords[-2][1]
        else:
            dx = 0
            dy = 0

        if gesture == "click":
            self.mouse.press()
            self.pressing = True
            self.mouse.move(coord[0], coord[1])
        elif gesture == "right-click":
            self.mouse.press(right=True)
            self.mouse.release(right=True)
            return 0, 0
        elif self.pressing:
            self.mouse.release(right=self.right_key)
            self.right_key = False
            self.pressing = False

        if gesture == "swipe" and len(gestures) >= 2 and gestures[-2] == "swipe":
            if dy > 0:
                self.mouse.scroll_down(dy)
            else:
                self.mouse.scroll_up(-dy)
            return 0, dy
        elif gesture == "point" and len(gestures) >= 2 and gestures[-2] == "point":
            self.mouse.move(coord[0], coord[1])
        return coord

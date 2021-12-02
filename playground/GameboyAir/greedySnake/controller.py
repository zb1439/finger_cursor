from finger_cursor.controller import Controller, CONTROLLER
from finger_cursor.driver import Keyboard


@CONTROLLER.register()
class GreedySnakeController(Controller):
    def __init__(self, cfg):
        super(GreedySnakeController, self).__init__(cfg)
        self.min_thresh = cfg.CONTROLLER.MIN_THRESH
        self.ratio = self.min_thresh[0] / self.min_thresh[1]
        self.min_thresh = self.min_thresh[0]
        self.keyboard = Keyboard()
        self.pressed = None

    def apply(self, gestures, coords):
        gestures = gestures[-3:]
        gesture = g1 = gestures[0]
        for g in gestures:
            if g != g1:
                gesture = "n/a"
                break

        if gesture != "n/a":
            self.press(gesture)
        return 0, 0

    def press(self, key):
        if self.pressed == key:
            return
        if self.pressed is not None:
            self.keyboard.release(self.pressed)

        self.pressed = key
        self.keyboard.press(key)

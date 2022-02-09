from finger_cursor.controller import Controller, CONTROLLER
from finger_cursor.driver import Keyboard

import time


@CONTROLLER.register()
class MarioController(Controller):
    def __init__(self, cfg):
        super(MarioController, self).__init__(cfg)
        self.keyboard = Keyboard()
        self.pressed = None
        self.gesture_to_key = {
                "left": "a", "right": "d"}

    def apply(self, gestures, coords):
        gestures = gestures[-3:]
        gesture = g1 = gestures[0]

        # Jump is usually a sudden gesture
        if gestures[-1] == "smalljump":
            # check if it is a new one
            self.keyboard.press("o")
            time.sleep(0.01)
            self.keyboard.release("o")
            return 0, 0

        elif gestures[-1] == "largejump":
            # check if it is a new one
            self.keyboard.press("o")
            time.sleep(0.3)
            self.keyboard.release("o")
            return 0, 0

        # A gesture must maintain at least 3 frames
        # otherwise, assign it to n/a
        for g in gestures:
            if g != g1:
                gesture = "n/a"
                break

        if gesture == "n/a":
            if self.pressed is not None:
                self.keyboard.release(self.pressed)
                self.pressed = None # reset
        else:
            self.press(self.gesture_to_key[gesture])
        return 0, 0

    def press(self, key):
        # same key, don't do anything
        if self.pressed == key:
            return
        if self.pressed is not None:
            self.keyboard.release(self.pressed)

        self.pressed = key
        self.keyboard.press(key)

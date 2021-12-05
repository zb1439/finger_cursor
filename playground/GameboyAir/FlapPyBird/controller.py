from finger_cursor.controller import Controller, CONTROLLER
from finger_cursor.driver import Keyboard


@CONTROLLER.register()
class FlappyBirdController(Controller):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.keyboard = Keyboard()

    def apply(self, gestures, coords):
        print(gestures)
        if len(gestures) >= 2 and gestures[-1] == "open":
            start = float("inf")
            for i in range(len(gestures) - 1):
                if gestures[i] == "fist":
                    start = i

            if start < len(gestures):
                all_na = True
                for i in range(start + 1, len(gestures) - 1):
                    if gestures[i] != "n/a":
                        all_na = False
                        break
                if all_na:
                    self.keyboard.press_and_release("space")

        return 0, 0

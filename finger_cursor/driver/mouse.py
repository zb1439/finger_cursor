from pynput import mouse


class Mouse:
    def click(self, right=False):
        raise NotImplementedError

    def double_click(self, right=False):
        raise NotImplementedError

    def press(self, right=False):
        raise NotImplementedError

    def release(self, right=False):
        raise NotImplementedError

    def move(self, x, y):
        raise NotImplementedError

    def move_rel(self, dx, dy):
        raise NotImplementedError

    def get_pos(self):
        return 0, 0

    def scroll_up(self, x):
        raise NotImplementedError

    def scroll_down(self, x):
        raise NotImplementedError


class WinMouse(Mouse):
    def __init__(self):
        self.ctrl = mouse.Controller()

    def click(self, right=False):
        self.ctrl.click(mouse.Button.right if right else mouse.Button.left)

    def double_click(self, right=False):
        self.ctrl.click(mouse.Button.right if right else mouse.Button.left, 2)

    def press(self, right=False):
        self.ctrl.press(mouse.Button.right if right else mouse.Button.left)

    def release(self, right=False):
        self.ctrl.release(mouse.Button.right if right else mouse.Button.left)

    def move(self, x, y):
        self.ctrl.position = (x, y)

    def move_rel(self, dx, dy):
        self.ctrl.move(dx, dy)

    def get_pos(self):
        return self.ctrl.position

    def scroll_up(self, x):
        self.ctrl.scroll(0, -x)

    def scroll_down(self, x):
        self.ctrl.scroll(0, x)


class MacMouse(WinMouse):
    def scroll_left(self, x):
        self.ctrl.scroll(-x, 0)

    def scroll_right(self, x):
        self.ctrl.scroll(x, 0)

    def swipe_up(self):  # TODO: i did not find how to implement this
        raise NotImplementedError

    def swipe_down(self):
        raise NotImplementedError

    def swipe_left(self):
        raise NotImplementedError

    def swipe_right(self):
        raise NotImplementedError

    def zoom_in(self, x):
        raise NotImplementedError

    def zoom_out(self, x):
        raise NotImplementedError

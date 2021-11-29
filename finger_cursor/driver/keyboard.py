from pynput.keyboard import Key, Controller


def str_to_key(func):
    def wrapper(obj, key):
        if hasattr(Key, key):
            key = getattr(Key, key)
        func(obj, key)

    return wrapper


class Keyboard:
    def __init__(self):
        self.ctrl = Controller()

    @str_to_key
    def press(self, key):
        self.ctrl.press(key)

    @str_to_key
    def release(self, key):
        self.ctrl.release(key)

    @str_to_key
    def press_and_release(self, key):
        self.ctrl.press(key)
        self.ctrl.release(key)

    @str_to_key
    def pressed(self, key):
        self.ctrl.pressed(key)

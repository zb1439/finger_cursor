from pynput.keyboard import Key, Controller


def str_to_key(func):
    def wrapper(obj, key):
        if hasattr(Key, key):
            key = getattr(Key, key)
        try:
            func(obj, key)
        except:
            raise ValueError(f"{key} does not exist in the key map")

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
        """
        Pressed is safe since it will finally release the key.
        """
        self.ctrl.pressed(key)

from inspect import signature
from pynput import keyboard
from pynput.keyboard import Key, Controller


def str_to_key(func):
    if len(str(signature(func)).split(',')) == 2:
        def wrapper(obj, key):
            if hasattr(Key, key):
                key = getattr(Key, key)
            try:
                rtn = func(obj, key)
                return rtn
            except:
                raise ValueError(f"{key} does not exist in the key map")
        return wrapper
    elif len(str(signature(func)).split(',')) == 1:
        def wrapper(key):
            if hasattr(Key, key):
                key = getattr(Key, key)
            try:
                rtn = func(key)
                return rtn
            except:
                raise ValueError(f"{key} does not exist in the key map")
        return wrapper
    else:
        raise ValueError(f"{func.__name__} must be a function or a class method with a single argument key, but got"
                         f"{str(signature(func))}")


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


keys = {}


def on_press(key):
    if key in keys.keys():
        keys[key] = True
    elif hasattr(key, 'char') and key.char in keys.keys():
        keys[key.char] = True


def on_release(key):
    pass


@str_to_key
def add_listener(key):
    if key not in keys:
        keys[key] = False


@str_to_key
def is_pressed(key):
    assert key in keys, f"{key} not in listener!"
    rtn = keys[key]
    keys[key] = False
    return rtn


listener = keyboard.Listener(on_press=on_press, on_release=on_release)

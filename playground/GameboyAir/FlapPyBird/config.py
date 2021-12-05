from finger_cursor.config import BaseConfig
from flappy import *  # noqa
from controller import *  # noqa
from classifier import *  # noqa


_config_dict = dict(
    APPLICATION=dict(
        NAME="FlappyBird",
        ASYNC=True,  # do not change this to false
    ),
    MODEL=dict(
        CLASSIFIER=dict(
            NAME="FlappyBirdClassifier",
            GESTURES=["n/a", "fist", "open"],
        ),
    ),
    CONTROLLER=dict(
        NAME="FlappyBirdController",
        HISTORY=5,
    ),
)


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        self._register_configuration(_config_dict)


config = Config()

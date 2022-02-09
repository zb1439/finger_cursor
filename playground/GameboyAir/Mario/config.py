from finger_cursor.config import BaseConfig
from mario import Mario  # noqa
from controller import MarioController  # noqa
from classifier import MarioClassifier  # noqa


_config_dict = dict(
    APPLICATION=dict(
        NAME="Mario",
        ASYNC=True,
    ),
    MODEL=dict(
        CLASSIFIER=dict(
            NAME="MarioClassifier",
            GESTURES=["n/a", "left", "right", "smalljump", "largejump"],
        ),
    ),
    CONTROLLER=dict(
        NAME="MarioController",
    ),
)


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        self._register_configuration(_config_dict)


config = Config()

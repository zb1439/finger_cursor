from finger_cursor.config import BaseConfig
from greedy_snake import *  # noqa
from controller import *  # noqa
from classifier import *  # noqa


_config_dict = dict(
    APPLICATION=dict(
        NAME="GreedySnake",
        ASYNC=True,
    ),
    MODEL=dict(
        CLASSIFIER=dict(
            NAME="GreedySnakeClassifier",
            GESTURES=["n/a", "up", "down", "left", "right"],
        ),
    ),
    CONTROLLER=dict(
        NAME="GreedySnakeController",
        MIN_THRESH=(0.01, 0.05),
        SCALE=(1, 1),
    ),
)


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        self._register_configuration(_config_dict)


config = Config()

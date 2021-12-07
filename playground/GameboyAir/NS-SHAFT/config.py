from finger_cursor.config import BaseConfig
from nsshaft import NSShaft  # noqa
from controller import NSShaftController  # noqa
from classifier import NSShaftClassifier  # noqa


_config_dict = dict(
    APPLICATION=dict(
        NAME="NSShaft",
        ASYNC=True,
    ),
    MODEL=dict(
        CLASSIFIER=dict(
            NAME="NSShaftClassifier",
            GESTURES=["n/a", "left", "right"],
        ),
    ),
    CONTROLLER=dict(
        NAME="NSShaftController",
    ),
)


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        self._register_configuration(_config_dict)


config = Config()

from finger_cursor.config import BaseConfig
from detector import *  # noqa
from evaluator import *  # noqa


_config_dict = dict(
    APPLICATION=dict(
        NAME="StaticVarianceEvaluator",
        ASYNC=False,
    ),
    DRIVER=dict(
        CAMERA=dict(
            NAME="VirtualCamera",
            VIDEO_PATH="static2.mp4",
        )
    ),
    MODEL=dict(
        DETECTOR=dict(
            NAME="FullPointKalmanDetector",
            # NAME="FullPointDetector",
        )
    ),
    CONTROLLER=dict(
        NAME="DummyController",
    )
)


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        self._register_configuration(_config_dict)


config = Config()

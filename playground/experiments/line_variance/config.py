from finger_cursor.config import BaseConfig
from detector import *  # noqa
from evaluator import *  # noqa


_config_dict = dict(
    APPLICATION=dict(
        NAME="LineVarianceEvaluator",
        ASYNC=False,
    ),
    DRIVER=dict(
        CAMERA=dict(
            NAME="VirtualCamera",
            VIDEO_PATH="line2.mov",
        )
    ),
    MODEL=dict(
        DETECTOR=dict(
            # NAME="FullPointKalmanDetector",
            NAME="FullPointDetector",
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

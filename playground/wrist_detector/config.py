from finger_cursor.config import BaseConfig
from wrist_detector import *  # noqa


_config_dict=dict(
    MODEL=dict(
        DETECTOR=dict(
            NAME="KalmanDetector",
            WRAPPED_NAME="WristDetector",
            MEASURE_NOISE=1.,
            PROCESS_NOISE=0.003,
        )
    )
)


class WristConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(_config_dict)


config = WristConfig()

from finger_cursor.config import BaseConfig
from threshold_control import *  # noqa


_config_dict=dict(
    CONTROLLER=dict(
        NAME="ThresholdRelativeController"
    )
)


class ThreshConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(_config_dict)


config = ThreshConfig()

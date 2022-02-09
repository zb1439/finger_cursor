from finger_cursor.config import BaseConfig
from camera import *  # noqa


_config_dict=dict(
    DRIVER=dict(
       CAMERA=dict(
           NAME="CollectingCamera",
           VIDEO_PATH="",
       ),
    ),
)


class WristConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(_config_dict)


config = WristConfig()

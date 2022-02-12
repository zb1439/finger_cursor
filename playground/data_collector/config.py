from finger_cursor.config import BaseConfig


_config_dict=dict(
    DRIVER=dict(
        CAMERA=dict(
            NAME="CollectingCamera",
            VIDEO_PATH="",
        ),
        CONTROLLER=dict(
            NAME="DummyController",
        )
    ),
)


class CollectingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(_config_dict)


config = CollectingConfig()

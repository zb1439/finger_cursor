from finger_cursor.config import BaseConfig


_config_dict=dict(
    MODEL=dict(
        CLASSIFIER=dict(
            NAME="MobileNetV2",
        )
    )
)


class MLPConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(_config_dict)


config = MLPConfig()

from finger_cursor.config import BaseConfig


_config_dict = dict(
    MODEL=dict(
        DETECTOR=dict(
            NAME="IndexTipDetector",
        )
    )
)


class WristConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(_config_dict)


config = WristConfig()

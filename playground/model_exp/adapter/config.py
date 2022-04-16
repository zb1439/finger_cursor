from finger_cursor.config import BaseConfig


_config_dict=dict(
    MODEL=dict(
        CLASSIFIER=dict(
            NAME="BlazeNet",
            ADAPTER=dict(
                ENABLE=True,
                SAMPLE_FRAMES=50,
                N_FRAMES=10,
                LR=1e-5,
                EPOCHS=2,
            ),
        )
    ),
    CONTROLLER=dict(
        NAME="DummyController",
    ),
)


class AdapterConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._register_configuration(_config_dict)


config = AdapterConfig()

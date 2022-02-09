import matplotlib
matplotlib.use("MacOSX")
import os
import sys
from finger_cursor.config import BaseConfig
from finger_cursor.engine import default_parser, main_process, merge_args_to_config

sys.path.append(os.getcwd())

try:
    from config import config
except ModuleNotFoundError:
    print("Warning: local config not found, using the default BaseConfig")
    config = BaseConfig()


if __name__ == '__main__':
    args = default_parser().parse_args()
    config = merge_args_to_config(args, config)
    print(config)
    main_process(config)

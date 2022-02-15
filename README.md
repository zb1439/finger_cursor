# Finger Cursor

This project provides a framework to control the mouse cursor and
raise mouse events by tracking hand keypoints and recognizing certain
gestures. 
This version just provides a simple demo 
(using [MediaPipe](https://google.github.io/mediapipe/) 
to get hand landmarks and classify gestures based on landmark coordinates) 
and basic utilities.

## Features

- Configurable
- Easy to modify (see playground examples)
- Unified pipeline for both deep learning and traditional CV methods
- Support Mac OS and Windows

## Setup

1. We recommed creating a separate virtual environment for this module.

2. Run `$python3 setup.py install`

    **_Note_**: Make sure to **rerun** the setup script everytime you checkout a new branch. Otherwise you may not be able to execute the right scripts.

## Usage

1. To run with default settings.

    ```bash
    cd <path_to_finger_cursor>/tools/
    bash finger_cursor
    ```

2. To run with data collector.

    ```bash
    cd <path_to_finger_cursor>/playground/data_collector/
    bash ../../tools/finger_cursor
    ```

     **_Note_**: You need to run the `finger_cursor` bash script from `data_collector` folder to use the config file in that folder. Otherwise the collecting camera will NOT be triggered.

3. To run with customized configuration.

    - Create a new folder `<somefolder>` under `/finger_cursor/playground/`.
    - Create your customized `config.py` inside `<somefolder>`. (See `/finger_cursor/config/base_config.py` as an example.)
    - If necessary, create customized modules that inherite from existing finger_cursor library according to your need. (See `/finger_cursor/playground/data_collector/` as an example.)
    - Run `/finger_cursor/tools/finger_cursor` bash script from `<somefolder>`. 

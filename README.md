# Finger Cursor

This project provides a framework to control the mouse cursor and
raise mouse events by tracking hand keypoints and recognizing certain
gestures. 
This version just provides a simple demo 
(using [MediaPipe](https://google.github.io/mediapipe/) 
to get hand landmarks and classify gestures based on landmark coordinates) 
and basic utilities.

## Features

- configurable
- easy to modify (see playground examples)
- unified pipeline for both deep learning and traditional CV methods
- Support Mac OS and Windows

## Install

`python3 setup.py install`

## Usage
For default settings, just run `finger_cursor`

Or you can create a new folder under `/playground`
with custom `config.py` and probably define some custom modules inherited from
finger cursor library and register them (see playground examples).

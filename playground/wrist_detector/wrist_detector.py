import cv2
import numpy as np

from finger_cursor.model import DETECTOR
from finger_cursor.model.detector.detector import Detector


@DETECTOR.register()
class WristDetector(Detector):
    def predict(self, features):
        landmarks = features["landmark"]
        if not landmarks.multi_hand_landmarks:
            return -1, -1
        landmarks = landmarks.multi_hand_landmarks[0].landmark
        return landmarks[0].x, landmarks[0].y

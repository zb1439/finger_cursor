import cv2
import numpy as np

from finger_cursor.model import DETECTOR
from finger_cursor.model.detector.detector import Detector


@DETECTOR.register()
class PalmCenterDetector(Detector):
    def predict(self, features):
        landmarks = features["landmark"]
        if not landmarks.multi_hand_landmarks:
            return -1, -1
        landmarks = landmarks.multi_hand_landmarks[0].landmark
        mean_x = np.mean([landmarks[i].x for i in [5, 9, 17]])
        mean_y = np.mean([landmarks[i].y for i in [5, 9, 17]])
        return mean_x, mean_y

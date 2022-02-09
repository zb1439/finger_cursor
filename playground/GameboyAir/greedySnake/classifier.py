import numpy as np

from finger_cursor.model import Classifier, CLASSIFIER


@CLASSIFIER.register()
class GreedySnakeClassifier(Classifier):
    def predict(self, features):
        landmarks = features["landmark"][-1]
        if not landmarks.multi_hand_landmarks:
            return 0

        landmarks = landmarks.multi_hand_landmarks[0].landmark
        # fingers = features["fingers"][-1]
        # is_fist = not np.any(fingers)

        dx = landmarks[0].x - np.mean([landmarks[i].x for i in [5, 9, 17]])
        dy = landmarks[0].y - np.mean([landmarks[i].y for i in [5, 9, 17]])
        theta = np.arctan(dy / dx) / np.pi * 180
        theta += (180 if dx < 0 else 0)

        if np.abs(theta - 90) < 15:
            return 1
        elif -90 <= theta < -60 or 260 < theta <= 270:
            return 2
        elif -20 <= theta <= 30:
            return 3
        elif 115 < theta <= 140:
            return 4

        return 0

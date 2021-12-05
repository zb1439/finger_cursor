import numpy as np

from finger_cursor.model import Classifier, CLASSIFIER


@CLASSIFIER.register()
class FlappyBirdClassifier(Classifier):
    def predict(self, features):
        landmarks = features["landmark"][-1]
        if not landmarks.multi_hand_landmarks:
            return 0

        landmarks = landmarks.multi_hand_landmarks[0].landmark
        finger_angles = [self._get_angle(landmarks, i) for i in range(5)]
        fingers = np.zeros(5, dtype=np.bool)
        fingers[0] = finger_angles[0] < 35
        fingers[1] = finger_angles[1] < 25
        fingers[2] = finger_angles[2] < 25
        fingers[3] = finger_angles[3] < 25
        fingers[4] = finger_angles[4] < 15
        if np.all(fingers[1:]):
            return 2
        elif np.all(~fingers[1:]):
            return 1
        else:
            return 0

    def _get_line(self, landmarks, idx1, idx2):
        return np.array([
            landmarks[idx1].x - landmarks[idx2].x,
            landmarks[idx1].y - landmarks[idx2].y,
            landmarks[idx1].z - landmarks[idx2].z
        ])

    def _get_segments(self, landmarks, finger_index):
        mcp_idx = finger_index * 4 + 1
        segm = [self._get_line(landmarks, 0, mcp_idx)]
        for i in range(3):
            segm.append(self._get_line(landmarks, mcp_idx + i, mcp_idx + i + 1))
        return segm

    def _compute_angle(self, line1, line2):
        cos = np.sum(line1 * line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
        theta = np.arccos(cos)
        return theta * 180 / np.pi

    def _get_angle(self, landmarks, finger_index):
        segm = self._get_segments(landmarks, finger_index)
        first_segm = segm[-1]
        angles = [self._compute_angle(first_segm, _segm) for _segm in segm[:-1]]
        return max(angles)

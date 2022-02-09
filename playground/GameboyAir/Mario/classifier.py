import numpy as np

from finger_cursor.model import Classifier, CLASSIFIER


@CLASSIFIER.register()
class MarioClassifier(Classifier):
    def predict(self, features):
        landmarks = features["landmark"][-1]
        fingers = features["fingers"][-1]

        # No hand
        if not landmarks.multi_hand_landmarks:
            return 0

        landmarks = landmarks.multi_hand_landmarks[0].landmark

        # Compute whether each finger is straight (by angles)
        fingers, finger_angles = self.get_fingers(landmarks)

        # [dx, dy]: a vector pointing to wrist
        dx = landmarks[0].x - np.mean([landmarks[i].x for i in [5, 9, 17]])
        dy = landmarks[0].y - np.mean([landmarks[i].y for i in [5, 9, 17]])

        theta = np.rad2deg(np.arctan2(dy, dx))

        # print(fingers, finger_angles)
        if np.sum(fingers) >= 3:
            return 4
        elif fingers[1] and fingers[2]:
            return 3
        # Move left
        elif -15 <= theta <= 75:
            return 1 # move (left)
        # Move right
        elif 105 <= theta or theta <= -165:
            return 2 # move (right)

        return 0

    def get_fingers(self, landmarks):
        finger_angles = [self._get_angle(landmarks, i) for i in range(5)]
        fingers = np.zeros(5, dtype=np.bool)
        fingers[0] = finger_angles[0] < 35
        fingers[1] = finger_angles[1] < 25
        fingers[2] = finger_angles[2] < 25
        fingers[3] = finger_angles[3] < 30
        fingers[4] = finger_angles[4] < 30
        return fingers, finger_angles

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

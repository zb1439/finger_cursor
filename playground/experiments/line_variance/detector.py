from finger_cursor.model.detector import DETECTOR, Detector, KalmanDetector


@DETECTOR.register()
class FullPointDetector(Detector):
    def predict(self, features):
        landmarks = features["landmark"]
        if not landmarks.multi_hand_landmarks:
            return -1, -1
        landmarks = landmarks.multi_hand_landmarks[0].landmark
        return [(landmarks[i].x, landmarks[i].y) for i in range(21)]


@DETECTOR.register()
class FullPointKalmanDetector(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kalman_filters = {}
        for i in range(21):
            cfg.MODEL.DETECTOR.TRACK_KEYPOINT = i
            self.kalman_filters[i] = KalmanDetector(cfg)

    def predict(self, features):
        return [self.kalman_filters[i].predict(features) for i in range(21)]

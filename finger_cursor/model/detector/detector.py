import cv2
import numpy as np

from finger_cursor.utils import queue, Registry


DETECTOR = Registry("DETECTOR")


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature = cfg.MODEL.DETECTOR.FEATURE
        self.feature = {t[0]: t[1] for t in self.feature}
        capacity = cfg.MODEL.DETECTOR.CAPACITY if hasattr(cfg.MODEL.DETECTOR, "CAPACITY") else -1
        self.queue = queue(self.__class__.__name__, capacity=capacity)

    def get_features(self):
        """
        :param features: list of dict from FeatureExtractor's queue
        :return: x, y
        """
        features = {}
        for k, v in self.feature.items():
            feat_q = queue(v)
            assert len(feat_q) > 0, f"feature {k} implemented by {v} not found in queue"
            features[k] = feat_q[-1]
        return features

    def predict(self, features):
        raise NotImplementedError

    def __call__(self):
        rtn = self.predict(self.get_features())
        self.queue.add(rtn)
        return rtn


@DETECTOR.register()
class KeypointDetector(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kp_idx = cfg.MODEL.DETECTOR.TRACK_KEYPOINT

    def predict(self, features):
        landmarks = features["landmark"]
        if not landmarks.multi_hand_landmarks:
            return -1, -1
        landmarks = landmarks.multi_hand_landmarks[0].landmark
        return landmarks[self.kp_idx].x, landmarks[self.kp_idx].y


@DETECTOR.register()
class IndexTipDetector(KeypointDetector):  # This class is maintained to support older version configs
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kp_idx = 8


@DETECTOR.register()
class KalmanDetector(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)
        cov_process = self.cfg.MODEL.DETECTOR.PROCESS_NOISE
        cov_measure = self.cfg.MODEL.DETECTOR.MEASURE_NOISE
        self.kf = cv2.KalmanFilter(6, 2, 0)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                             [0, 1, 0, 1, 0, 0.5],
                                             [0, 0, 1, 0, 1, 0],
                                             [0, 0, 0, 1, 0, 1],
                                             [0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * cov_process
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * cov_measure
        self.wrapped_detector = DETECTOR.get(self.cfg.MODEL.DETECTOR.WRAPPED_NAME)(cfg)

    def predict(self, features):
        x, y = self.wrapped_detector.predict(features)
        if x == -1:  # not enabled
            return x, y
        measure = np.array([x, y], dtype=np.float32)
        self.kf.correct(measure)
        filtered = self.kf.predict()
        return filtered[0].item(), filtered[1].item()


import numpy as np

from finger_cursor.utils import queue, Registry


CLASSIFIER = Registry("CLASSIFIER")


class Classifier:
    def __init__(self, cfg):
        self.feature = cfg.MODEL.CLASSIFIER.FEATURE
        self.feature = {t[0]: t[1] for t in self.feature}
        self.history = cfg.MODEL.CLASSIFIER.HISTORY
        self.gestures = {i: name for i, name in enumerate(cfg.MODEL.CLASSIFIER.GESTURES)}
        capacity = cfg.MODEL.CLASSIFIER.CAPACITY if hasattr(cfg.MODEL.CLASSIFIER, "CAPACITY") else -1
        self.queue = queue(self.__class__.__name__, capacity=capacity)

    def get_features(self):
        features = {}
        for k, v in self.feature.items():
            feat_q = queue(v)
            assert len(feat_q) > 0, f"feature {k} implemented by {v} not found in queue"
            features[k] = feat_q[-self.history:]
        return features

    def predict(self, features):
        """
        :param features: list of dict from FeatureExtractor's queue
        :return: an index indicating which gesture (the index will be mapped to a string representing the
                 gesture automatically).
        """
        raise NotImplementedError

    def __call__(self):
        pred_idx = self.predict(self.get_features())
        rtn = self.gestures[pred_idx]
        self.queue.add(rtn)
        return rtn


@CLASSIFIER.register()
class RuleClassifier(Classifier):
    def predict(self, features):
        landmarks = features["landmark"][-1]
        if not landmarks.multi_hand_landmarks:
            return 0
        landmarks = landmarks.multi_hand_landmarks[0].landmark

        # Voting for fingers
        fingers = features["fingers"]  # [[5], [5], ..., [5]] list of t fingers
        fingers = np.stack(fingers, 0)  # stack over time dimension
        fingers = np.sum(fingers, 0) > (len(fingers) // 2)

        def distance(pt1, pt2):
            return np.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

        pick_dist = distance(landmarks[4], landmarks[8]) / (distance(landmarks[12], landmarks[0]) + 1e-5)
        swipe_dist = np.mean([distance(landmarks[i], landmarks[i+4]) for i in [6, 7, 8]]) \
                     / (distance(landmarks[8], landmarks[0]) + 1e-5)
        if not np.any(fingers):  # no straight finger (fist)
            return 0
        if pick_dist <= 0.09:  # thumb tip to index tip smaller than middle tip to wrist (ok)
            return 1
        if fingers[1] and fingers[2] and swipe_dist <= 0.09 and swipe_dist < pick_dist \
                and not fingers[0] and not fingers[3] and not fingers[4]:
            # distances between index finger keypoints and middle finger keypoints
            # less than the distance from index finger tip to wrist (victory w/ two finger closed, click)
            return 4
        if np.all(fingers):  # open hand
            return 5
        if fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:  # victory
            return 3
        if fingers[1] and not fingers[2] and not fingers[0]:  # point
            return 2
        return 0

from argparse import Namespace
import cv2
import mediapipe as mp
import numpy as np

from finger_cursor.utils import queue, Registry


FEATURE_EXTRACTOR = Registry("FEATURE_EXTRACTOR")
mp_hands = mp.solutions.hands


class FeatureExtractor:
    def __init__(self, all_config, visited):
        self.cfg = Namespace(**(all_config[self.__class__.__name__][0]))
        if not hasattr(self.cfg, "capacity"):
            setattr(self.cfg, "capacity", -1)
        self.queue = queue(self.__class__.__name__, self.cfg.capacity)
        self.dependent_names = all_config[self.__class__.__name__][1]
        self.dependencies = {}
        for k, v in self.dependent_names.items():
            if v in visited:
                continue
            visited.append(v)
            self.dependencies[k] = FEATURE_EXTRACTOR.get(v)(all_config, visited)


    def apply(self, image, extra_info):
        """
        :param image: original image (probably preprocessed)
        :param extra_info: extra_info from preprocessors
        :return: any variable representing the feature
        """
        raise NotImplementedError

    def __call__(self, image, extra_info):
        for v in self.dependencies.values():
            v(image, extra_info)
        feature = self.apply(image, extra_info)
        self.queue.add(feature)


class FeatureExtractorGraph:
    def __init__(self, cfg):
        configs = cfg.MODEL.FEATURE_EXTRACTOR
        all_config = dict()
        roots = [c[0] for c in configs]
        for cfg in configs:
            assert cfg[0] not in all_config
            all_config[cfg[0]] = cfg[1:]
            for child in cfg[2].values():
                roots.remove(child)

        self.roots = [FEATURE_EXTRACTOR.get(name)(all_config, roots) for name in roots]

    def apply(self, image, extra_info):
        for root in self.roots:
            root(image, extra_info)


@FEATURE_EXTRACTOR.register()
class MediaPipeHandLandmark(FeatureExtractor):
    def __init__(self, all_config, visited):
        super().__init__(all_config, visited)
        self.model = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.7,
                                    min_tracking_confidence=0.5)

    def apply(self, image, extra_info):
        result = self.model.process(image[..., ::-1])
        return result


@FEATURE_EXTRACTOR.register()
class FingerDescriptor(FeatureExtractor):  # rule-based finger indicator, dependent on finger landmarks
    def apply(self, image, extra_info):
        fingers = np.zeros(5, dtype=np.bool)
        features = queue(self.dependent_names["landmark"])[-1]
        if not features.multi_hand_landmarks:
            return fingers

        landmark = features.multi_hand_landmarks[0].landmark
        center_point_x = sum([landmark[5].x, landmark[9].x, landmark[17].x]) / 3
        center_point_y = sum([landmark[5].y, landmark[9].y, landmark[17].y]) / 3
        normalizer = np.sqrt((center_point_x - landmark[0].x) ** 2 + (center_point_y - landmark[0].y) ** 2)

        fingers[0] = self.thumb_angle(landmark) <= 11 or abs(landmark[4].x - landmark[0].x) >= 0.05
        fingers[1] = 0.45 <= self.finger_ratio(landmark, 5) <= 1.0
        fingers[2] = 0.45 <= self.finger_ratio(landmark, 9) <= 1.0 and self.distance(landmark[12], landmark[0]) >= 0.25
        fingers[3] = self.distance(landmark[16], landmark[0]) >= 0.28
        fingers[4] = self.distance(landmark[20], landmark[0]) >= 0.2
        return fingers

    def distance(self, pt1, pt2, normalizer=None):
        raw_distance = np.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)
        if normalizer is not None:
            return raw_distance / (normalizer + 1e-5)
        return raw_distance

    def finger_ratio(self, landmark, base):
        return self.distance(landmark[base+3], landmark[base]) / (self.distance(landmark[base+3], landmark[0]) + 1e-5)

    def thumb_angle(self, landmark):
        wrist = landmark[mp_hands.HandLandmark.WRIST]
        mcp = landmark[mp_hands.HandLandmark.THUMB_MCP]
        tip = landmark[mp_hands.HandLandmark.THUMB_TIP]
        dir = np.array([mcp.x - wrist.x, mcp.y - wrist.y]) * 100
        thumb = np.array([tip.x - mcp.x, tip.y - mcp.y]) * 100
        theta = np.arccos(np.sum(dir * thumb) / (np.linalg.norm(dir) + 1e-5) / (np.linalg.norm(thumb) + 1e-5)) / np.pi * 180
        theta = 0 if np.isnan(theta) else theta
        return theta


@FEATURE_EXTRACTOR.register()
class RotationDescriptor(FeatureExtractor):  # TODO
    def apply(self, image, extra_info):
        raise NotImplementedError

import numpy as np
import os.path as osp
import pickle

from .classifier import CLASSIFIER, Classifier


@CLASSIFIER.register()
class MLP(Classifier):
    def __init__(self, cfg):
        super(MLP, self).__init__(cfg)
        self.cls_mapping = {
            0: 2,
            1: 0,
            2: 1,
            3: 4,
            4: 5,
            5: 0,
            6: 3
        }
        model_path = osp.join(osp.dirname(__file__), 'mlp_classifier.pickle')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, features):
        landmarks = features["landmark"][-1]
        if not landmarks.multi_hand_landmarks:
            return 0
        landmarks = landmarks.multi_hand_landmarks[0].landmark
        data = []
        for landmark in landmarks:
            data.append(landmark.x)
            data.append(landmark.y)
            data.append(landmark.z)
        data = np.array(data)[None]
        pred = self.model.predict(data)[0].item()
        return self.cls_mapping[pred]

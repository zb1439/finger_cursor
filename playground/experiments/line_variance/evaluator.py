import numpy as np

from finger_cursor.engine import Application, APPLICATION
from finger_cursor.utils import queue


@APPLICATION.register()
class LineVarianceEvaluator(Application):
    def __init__(self, cfg):
        super(LineVarianceEvaluator, self).__init__(cfg)
        # 21 keypoints
        self.point_seq = {i: [] for i in range(21)}
        self.det_queue = queue(cfg.MODEL.DETECTOR.NAME)

    def terminate(self):  # compute metrics here
        index_var = self.compute_var(8)
        thumb_var = self.compute_var(4)
        wrist_var = self.compute_var(0)
        print("index: {:2.4f}, thumb: {:2.4f}, wrist: {:2.4f}".format(
            index_var, thumb_var, wrist_var))

    def loop(self):
        points = self.det_queue[-1]
        for i in range(21):
            self.point_seq[i].append(points[i])

    def compute_var(self, landmark_idx):
        # INDEX TIP, THUMB IP, WRIST
        # (n, 2)
        X = np.array(self.point_seq[landmark_idx])
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        # ((singular value)^2) / n is the second principle component's variance
        n = X.shape[0]
        return (s[1] ** 2) / n


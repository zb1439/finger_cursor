import numpy as np

from finger_cursor.engine import Application, APPLICATION
from finger_cursor.utils import queue


@APPLICATION.register()
class StaticVarianceEvaluator(Application):
    def __init__(self, cfg):
        super(StaticVarianceEvaluator, self).__init__(cfg)
        self.point_seq = {i: [] for i in range(21)}
        self.det_queue = queue(cfg.MODEL.DETECTOR.NAME)

    def terminate(self):  # compute metrics here
        variance_per_pt = [np.var(self.point_seq[i]) for i in range(21)]
        print("{:2.4f}".format(np.mean(variance_per_pt)))

    def loop(self):
        points = self.det_queue[-1]
        for i in range(21):
            self.point_seq[i].append(points[i])

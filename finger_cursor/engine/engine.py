import argparse

from finger_cursor.controller import CONTROLLER
from finger_cursor.driver import CAMERA
from finger_cursor.model import CLASSIFIER, DETECTOR, FeatureExtractorGraph
from finger_cursor.preprocessor import PreprocessorStack
from finger_cursor.utils import AdaptingException, ExitException, queue
from finger_cursor.window import imshow, draw_mediapipe


def default_parser():
    parser = argparse.ArgumentParser("finger cursor prototyping")
    return parser


def main_process(cfg):
    camera = CAMERA.get(cfg.DRIVER.CAMERA.NAME)(cfg).stream()
    preproc = PreprocessorStack(cfg)
    feature_graph = FeatureExtractorGraph(cfg)
    cls = CLASSIFIER.get(cfg.MODEL.CLASSIFIER.NAME)(cfg)
    det = DETECTOR.get(cfg.MODEL.DETECTOR.NAME)(cfg)
    ctrl = CONTROLLER.get(cfg.CONTROLLER.NAME)(cfg)
    draw_landmark = cfg.VISUALIZATION.LANDMARK
    while True:
        try:
            frame = next(camera)
            try:
                frame, extra_info = preproc(frame, {})
                feature_graph.apply(frame, extra_info)
                feature = queue("MediaPipeHandLandmark")[-1]
                fingers = queue("FingerDescriptor")[-1]
                cls()
                det()
                gesture, coord = ctrl()
                if draw_landmark:
                    draw_mediapipe(frame, feature, text=gesture + "({:2.1f}, {:2.1f})".format(coord[0], coord[1]))
                else:
                    imshow(frame, text=gesture + "({:2.1f}, {:2.1f})".format(coord[0], coord[1]))
            except AdaptingException:
                pass
        except ExitException:
            break

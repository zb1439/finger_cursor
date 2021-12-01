import argparse

from finger_cursor.controller import CONTROLLER
from finger_cursor.driver import CAMERA
from finger_cursor.model import CLASSIFIER, DETECTOR, FeatureExtractorGraph
from finger_cursor.preprocessor import PreprocessorStack
from finger_cursor.utils import AdaptingException, ExitException, queue
from finger_cursor.window import imshow, draw_mediapipe
from .application import APPLICATION


def default_parser():
    parser = argparse.ArgumentParser("finger cursor prototyping")
    parser.add_argument("-s", "--show-window", action="store_true")
    return parser


def merge_args_to_config(args, config):
    if args.show_window:
        config.VISUALIZATION.SHOW_WINDOW = True
    return config


def main_process(cfg):
    camera = CAMERA.get(cfg.DRIVER.CAMERA.NAME)(cfg).stream()
    preproc = PreprocessorStack(cfg)
    feature_graph = FeatureExtractorGraph(cfg)
    cls = CLASSIFIER.get(cfg.MODEL.CLASSIFIER.NAME)(cfg)
    det = DETECTOR.get(cfg.MODEL.DETECTOR.NAME)(cfg)
    ctrl = CONTROLLER.get(cfg.CONTROLLER.NAME)(cfg)
    app = APPLICATION.get(cfg.APPLICATION.NAME)(cfg)
    draw_landmark = cfg.VISUALIZATION.LANDMARK
    show_window = cfg.VISUALIZATION.SHOW_WINDOW
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
                app.loop()
                if show_window:
                    if draw_landmark:
                        draw_mediapipe(frame, feature, text=gesture + "({:2.1f}, {:2.1f})".format(coord[0], coord[1]))
                    else:
                        imshow(frame, text=gesture + "({:2.1f}, {:2.1f})".format(coord[0], coord[1]))
            except AdaptingException:
                pass
        except ExitException:
            app.terminate()
            break

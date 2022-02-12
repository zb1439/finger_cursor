import cv2
from datetime import datetime
import json
import keyboard
import os
import sys

sys.path.append('/Users/Skye/finger_cursor/')
from finger_cursor.utils import CameraException, ExitException, Registry, queue, data_collection

CAMERA = Registry("CAMERA")


def stream(capturer,
           freq=100,
           on_exit=27,
           on_capture=32,
           max_run=-1,
           capture_callback=None,
           exit_callback=None,
           max_error=10):
    """
    :param capturer: video capture or other video frames (probably we need to wrap a video reader?)
    :param freq: time interval for capturing the next image
    :param on_exit: key for exit (default: esc)
    :param on_capture: key for capture (default: space)
    :param max_run: if max_run > 0, we will run the loop at most max_run times
    :param capture_callback: callback func when you press the capture key
    :param exit_callback: callback func when you press exit
    :param max_error: maximum number to try to get the next frame
    :return: a python generator, you get your image by `gen = stream(...); img = next(gen)`
    """
    error_count = 0
    frame_count = 0

    while True:
        got_image, frame = False, None
        while not got_image and (error_count < max_error or max_error == -1):
            error_count += 1
            got_image, frame = capturer.read()
            if not got_image and max_error == -1:
                break

        if max_error == -1 and not got_image:
            raise ExitException

        if error_count >= max_error and not got_image:
            raise CameraException("Connection to the camera failed after 10 trials")

        error_count = 0
        assert frame is not None

        try:
            frame = frame[:, ::-1]
        except:
            raise ValueError("Error when flipping the frame, check if you really got a frame")
        yield frame

        cv2.waitKey(1)
        if keyboard.is_pressed('q'):
            break
        elif keyboard.is_pressed('c') and capture_callback is not None:
            print('Capturing frame.')
            capture_callback(frame, frame_count)

        frame_count += 1
        if max_run > 0 and frame_count > max_run:
            break

    if exit_callback is not None:
        exit_callback()
    raise ExitException


class Camera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.freq = 1000 // cfg.DRIVER.CAMERA.FPS
        self.on_exit = cfg.DRIVER.CAMERA.ON_EXIT
        self.device_id = cfg.DRIVER.CAMERA.DEVICE_INDEX
        self.cap = self.setup()

    def setup(self):
        """
        :return: a video capturer with method read() returns the success code and the current frame
        """
        raise NotImplementedError

    def capture_callback(self):
        """
        :return: a function which takes an image as input and dump all necessary info inside the function
        e.g.,
            def foo(image):
             cv2.save('xxx.jpg', image)
             feature = queue('MediaPipeHandLandmark')[-1]
             np.save('xxx.npy', feature)
            return foo
        """
        return None

    def exit_callback(self):
        return None


@CAMERA.register()
class DefaultCamera(Camera):
    def setup(self):
        return cv2.VideoCapture(self.device_id)

    def stream(self):
        return stream(self.cap, self.freq, self.on_exit, on_capture=32,  # 32 is for spacebar
                      capture_callback=self.capture_callback(), exit_callback=self.exit_callback())


@CAMERA.register()
class VirtualCamera(DefaultCamera):
    def __init__(self, cfg):
        self.video_path = cfg.DRIVER.CAMERA.VIDEO_PATH
        super().__init__(cfg)

    def setup(self):
        return cv2.VideoCapture(self.video_path)

    def stream(self):
        return stream(self.cap, self.freq, self.on_exit, on_capture=32,  # 32 is for spacebar
                      capture_callback=self.capture_callback(), exit_callback=self.exit_callback(), max_error=-1)


@CAMERA.register()
class CollectingCamera(DefaultCamera):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.class_label = data_collection.get_label_class()
        root_path = os.getcwd()
        self.username = root_path.split('/')[2]
        self.img_path, self.label_path = data_collection.mkdirs(root_path, self.class_label)

    def capture_callback(self):
        img_path, label_path = self.img_path, self.label_path
        prefix = self.username + datetime.now().strftime("%Y%m%d%H%M%S")

        def func(img, frame_count):
            print('Capturing image', frame_count)
            print('Saving to', img_path)
            img_name = os.path.join(img_path, prefix + '_' + str(frame_count) + '.png')
            cv2.imwrite(img_name, img)

            gt = {'multi_hand_landmarks': [],
                  'multi_hand_world_landmarks': [],
                  'multi_handedness': []}
            feature = queue("MediaPipeHandLandmark")[-1]

            for data_pt in feature.multi_hand_landmarks:
                keypoints = [{'x': info.x, 'y': info.y, 'z': info.z} \
                             for info in data_pt.landmark]
                gt['multi_hand_landmarks'] += keypoints

            for data_pt in feature.multi_hand_world_landmarks:
                keypoints = [{'x': info.x, 'y': info.y, 'z': info.z} \
                             for info in data_pt.landmark]
                gt['multi_hand_world_landmarks'] += keypoints

            gt['multi_handedness'] = [{'index': info.index, 'score': info.score, 'label': info.label} \
                                      for info in feature.multi_handedness[0].classification]

            with open(os.path.join(label_path, prefix + '_' + str(frame_count) + '.json'), 'w') as f:
                json.dump(gt, f, indent=4)

        return func

import cv2
from datetime import datetime
import json
import os

from .keyboard import add_listener, is_pressed, listener
from finger_cursor.utils import CameraException, ExitException, Registry, queue, data_collection

CAMERA = Registry("CAMERA")


def stream(capturer,
           on_exit='q',
           on_capture='c',
           max_run=-1,
           capture_callback=None,
           exit_callback=None,
           max_error=10):
    """
    :param capturer: video capture or other video frames (probably we need to wrap a video reader?)
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
    add_listener(on_capture)
    add_listener(on_exit)

    listener_start = False  # pynput listener has to start after first call on cv2.waitKey (reason unknown)

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
        if not listener_start:
            listener.start()
            listener_start = True

        if is_pressed(on_exit):
            print('Program exit.')
            break
        elif capture_callback is not None and is_pressed(on_capture):
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
        return stream(self.cap, self.on_exit, on_capture='c',
                      capture_callback=self.capture_callback(), exit_callback=self.exit_callback())


@CAMERA.register()
class VirtualCamera(DefaultCamera):
    def __init__(self, cfg):
        self.video_path = cfg.DRIVER.CAMERA.VIDEO_PATH
        super().__init__(cfg)

    def setup(self):
        return cv2.VideoCapture(self.video_path)

    def stream(self):
        return stream(self.cap, self.on_exit, on_capture='c',
                      capture_callback=self.capture_callback(), exit_callback=self.exit_callback(), max_error=-1)


@CAMERA.register()
class CollectingCamera(DefaultCamera):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.class_label = data_collection.get_label_class()
        root_path = os.getcwd()
        self.username = data_collection.get_username()
        self.img_path, self.label_path = data_collection.mkdirs(root_path, self.class_label)

    def capture_callback(self):
        img_path, label_path = self.img_path, self.label_path
        prefix = self.username + datetime.now().strftime("%Y%m%d%H%M%S")

        def func(img, frame_count):
            img_name = os.path.join(img_path, prefix + '_' + str(frame_count) + '.png')

            gt = {'multi_hand_landmarks': [],
                  'multi_hand_world_landmarks': [],
                  'multi_handedness': []}
            feature = queue("MediaPipeHandLandmark")[-1]
            if feature is None:
                return

            print('Capturing image', frame_count)
            print('Saving to', img_path)
            cv2.imwrite(img_name, img)

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


class Sol:
    def __init__(self):
        self.duplicates = set()
        self.nondup = set()

    def countNumbers(self, arr):
        for lst in arr:
            count = 0
            start = lst[0]
            end = lst[1]
            for i in range(start, end + 1):
                if i in self.duplicates:
                    count += 1
                    continue
                if self.check(i):
                    self.duplicates.add(i)
                    count += 1;
            print(end - start + 1 - count)

    def check(self, num):
        seen = [False for _ in range(10)]
        while num > 0:
            n = num % 10
            if seen[n]:
                return True
            seen[n] = True
            num /= 10
        return False

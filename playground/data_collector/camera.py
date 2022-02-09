from finger_cursor.driver.camera import DefaultCamera, CAMERA
from finger_cursor.utils import queue


@CAMERA.register()
class CollectingCamera(DefaultCamera):
    def capture_callback(self):
        def func(img):
            print(img.shape)
            feature = queue("MediaPipeHandLandmark")[-1]
            print(feature)
            # TODO: transform the feature (Mediapipe SolutionOutputs instance to a numpy array)
            #  and save the images and the array
        return func

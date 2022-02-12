from itertools import count
import os
import cv2
import json
from finger_cursor.driver.camera import DefaultCamera, CAMERA
from finger_cursor.utils import queue


@CAMERA.register()
class CollectingCamera(DefaultCamera):
    def capture_callback(self):
        def func(img, frame_count, img_path, label_path, username):
            print('Capturing image', frame_count)
            print('Saving to', img_path)
            img_name = os.path.join(img_path, username+'_'+str(frame_count)+'.png')
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

            with open(os.path.join(label_path, username+'_'+str(frame_count)+'.json'), 'w') as f:
                json.dump(gt, f, indent=4)

        return func

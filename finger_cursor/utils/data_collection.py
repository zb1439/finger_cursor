import cv2
from datetime import datetime
import json
import os
import pwd


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(root_path, class_label):
    folder_path = os.path.join(root_path, class_label)
    img_path = os.path.join(folder_path, 'images')
    label_path = os.path.join(folder_path, 'labels')

    if not os.path.exists(folder_path): 
        os.mkdir(folder_path)
        os.mkdir(img_path)
        os.mkdir(label_path)

    return img_path, label_path


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def get_label_class():
    input_msg = '''
            Choose a label class to run data collection:
                0 - Palm
                1 - Finger point
                2 - Index finger pick
                3 - middle finger pick
                4 - Victory
                5 - Thumb up
                6 - Fist
                '''

    label_class_dict = {
        0: 'palm',
        1: 'finger_point',
        2: 'index_pick',
        3: 'middle_pick',
        4: 'victory',
        5: 'thumb_up',
        6: 'fist'
    }

    label_idx = input(input_msg)
    try:
        label = label_class_dict[int(label_idx)]
    except:
        print('***Expecting an integer input from 0 to 6!***')
        raise

    return label


def dump_data(img, feature, img_path, label_path, frame_count):
    prefix = get_username() + datetime.now().strftime("%Y%m%d%H%M%S")
    img_name = os.path.join(img_path, prefix + '_' + str(frame_count) + '.png')
    print(f"Capturing image {frame_count if frame_count >= 0 else ''}")
    print('Saving to', img_name)
    cv2.imwrite(img_name, img)

    gt = {'multi_hand_landmarks': [],
          'multi_hand_world_landmarks': [],
          'multi_handedness': []}
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


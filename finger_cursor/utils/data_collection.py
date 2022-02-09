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

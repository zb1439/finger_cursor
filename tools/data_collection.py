import cv2
import keyboard
import os
from cv2 import VideoCapture


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture(camera, save_dir):
    '''Capture images from camera and save to save_dir.
    
    Args:
        camera: opencv VideoCapture project.
        save_dir (str): Folder path to save the captured images.
    '''
    # fps = camera.get(cv2.CV_CAP_PROP_FPS)
    # fps = camera.get(5)
    # print('Started capturing at fps:', fps)
    count = 0
    while True:
        result, image = camera.read()

        if result:
            cv2.imshow('sample_image', image)
            ch = cv2.waitKey(1) & 0xFF
            if keyboard.is_pressed('q'):
                print('Ending capturing...')
                break
            cv2.destroyWindow('sample_image')

            img_name = os.path.join(save_dir, str(count)+'.png')
            print('Writing image to', img_name)
            # cv2.imwrite(img_name, image)
            count += 1
        else:
            print("No image detected.")


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

if __name__ == '__main__':

    cam_port = 0
    camera = VideoCapture(cam_port)
    label = get_label_class()
    root_path = '/Users/Skye/Desktop/data_collection/'
    folder_path = os.path.join(root_path, label)
    mkdir(folder_path)

    capture(camera, folder_path)

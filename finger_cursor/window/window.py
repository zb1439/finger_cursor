# drawing utilities

import cv2
import mediapipe as mp
import numpy as np
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles


cv2.namedWindow("demo")
windowNames = ["demo"]
image_buffer = dict()


start = time.time()
frames = 0


def _normalize_and_scale(image, method):
    if method == "rescale":
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif method == "clip":
        image = np.clip(image, 0, 1)
    else:
        raise NotImplementedError(f"{method} for normalization does not exist")
    return image


def notice(text, name=None, t=1000):
    image = np.zeros((256, 256), dtype=np.uint8)
    image = cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [256, 256, 256], 2)
    cv2.imshow(name or "demo", image)
    cv2.waitKey(t)


def imshow(image, name=None, text=None, normalize="rescale", show=True):
    if image.dtype in [np.float, np.float32, np.float64]:
        image = _normalize_and_scale(image, normalize)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if text is not None:
        colors = np.sort(image[:50, :50].reshape(-1, 3))
        color = colors[len(colors) // 2]
        color = [255 - color[0], 255 - color[1], 255 - color[2]]
        color = [int(c) for c in color]
        cv2.putText(image, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    global frames, start
    frames += 1
    fps = (time.time() - start) * 1000 / frames

    cv2.putText(image, "FPS:{:3.1f}".format(fps), (image.shape[1] - 100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if show:
        if name is not None and name not in windowNames:
            cv2.namedWindow(name)
            windowNames.append(name)
        cv2.imshow(name or "demo", image)
    else:
        return image


def imshow_later(image, name=None, text=None, normalize="rescale"):
    name = name or "demo"
    if name not in image_buffer:
        image_buffer[name] = []
    image_buffer[name].append(imshow(image, name, text, normalize, show=False))


def show(name="demo", destroy=False):
    assert name in windowNames, f"Window {name} not found"
    if not image_buffer.get(name, []):
        raise ValueError("Image buffer is empty")
    try:
        imshow(np.concatenate(image_buffer[name]), name)
    finally:
        image_buffer[name] = []
        if destroy:
            cv2.destroyWindow(name)


def draw_mediapipe(image, feature, name=None, text=None, normalize="rescale", show=True):
    if not feature.multi_hand_landmarks:
        imshow(image, name, "no hand", normalize, show)
        return

    landmark = feature.multi_hand_landmarks[0]
    def distance(pt1, pt2):
        return np.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)
    # print(distance(landmark.landmark[4], landmark.landmark[8]) / distance(landmark.landmark[12], landmark.landmark[0]))
    # print(np.mean([distance(landmark.landmark[i], landmark.landmark[i+4]) for i in [6, 7, 8]]) / distance(landmark.landmark[8], landmark.landmark[0]))
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, landmark, mp_hands.HAND_CONNECTIONS,
                              mp_drawing_style.get_default_hand_landmarks_style(),
                              mp_drawing_style.get_default_hand_connections_style())
    imshow(annotated_image, name, text, normalize, show)

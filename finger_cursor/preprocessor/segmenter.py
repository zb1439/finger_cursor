import cv2
import numpy as np

from finger_cursor.utils import AdaptingException
from finger_cursor.window import imshow

from .preprocessor import Preprocessor, PREPROCESSOR


@PREPROCESSOR.register()
class GMM(Preprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.frame_count = 0
        self.warm_up = 10
        self.model = cv2.createBackgroundSubtractorMOG2(self.cfg.history, self.cfg.init_var, False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.morph_ks, self.cfg.morph_ks))

    def process_info(self, image, extra_info):
        if self.frame_count < self.warm_up:
            imshow(image, text="modeling background")
            self.frame_count += 1
            raise AdaptingException
        elif self.frame_count < self.warm_up + self.cfg.history:
            self.model.apply(image)
            imshow(image, text="modeling background")
            self.frame_count += 1
            raise AdaptingException
        else:
            self.model.setVarThreshold(self.cfg.var)
            mask = self.model.apply(image, learningRate=0)
            mask = cv2.erode(mask, self.kernel, iterations=1)
            mask = cv2.dilate(mask, self.kernel, iterations=2)
            mask = mask / 255
            extra_info["GMM"] = mask
            return extra_info


@PREPROCESSOR.register()
class SkinSegm(Preprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.morph_ks, self.cfg.morph_ks))
        self.HSV_lo = self.cfg.hsv_lo
        self.HSV_hi = self.cfg.hsv_hi

    def process_info(self, image, extra_info):
        if "GMM" in extra_info:
            mask = extra_info["GMM"]
            masked_image = (image * mask[..., None]).astype(np.uint8)
        else:
            masked_image = image

        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.HSV_lo, self.HSV_hi)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        extra_info["SkinSegm"] = mask / 255
        return extra_info



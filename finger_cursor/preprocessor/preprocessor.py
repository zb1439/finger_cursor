from argparse import Namespace
import cv2
import numpy as np
import skimage.filters as filters
from skimage.morphology import disk

from finger_cursor.utils import conv, gaussian_kernel, Registry


PREPROCESSOR = Registry("PREPROCESSOR")


def get_preprocessor(name, config):
    return PREPROCESSOR.get(name)(config)


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = Namespace(**cfg)

    def process_image(self, image, extra_info):
        return image

    def process_info(self, image, extra_info):
        return extra_info

    def __call__(self, image, extra_info):
        return self.process_image(image, extra_info), self.process_info(image, extra_info)


class PreprocessorStack:
    def __init__(self, cfg):
        self.modules = []
        for name, config in cfg.PREPROCESS.PIPELINE:
            self.modules.append(get_preprocessor(name, config))

    def __call__(self, image, extra_info=None):
        extra_info = extra_info or {}
        for mod in self.modules:
            image, extra_info = mod(image, extra_info)
        return image, extra_info


@PREPROCESSOR.register()
class Resize(Preprocessor):
    def process_image(self, image, extra_info):
        return cv2.resize(image, (self.cfg.w, self.cfg.h))


@PREPROCESSOR.register()
class GaussianBlur(Preprocessor):
    def __init__(self, cfg):
        super(GaussianBlur, self).__init__(cfg)
        self.kernel = gaussian_kernel(self.cfg.size, self.cfg.sigma)

    def process_image(self, image, extra_info):
        return conv(image, self.kernel)


@PREPROCESSOR.register()
class MedianBlur(Preprocessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kernel = disk(self.cfg.size)

    def process_image(self, image, extra_info):
        return np.stack([filters.median(ch, self.kernel) for ch in cv2.split(image)], -1)

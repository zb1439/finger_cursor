import cv2
import numpy as np
from scipy.signal import convolve2d


def conv(img, kernel, **kwargs):
    """
    Wrapper function for convolution on rgb images w/ single channel filter.
    """
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        return np.stack([convolve2d(x, kernel, **kwargs) for x in [b, g, r]], -1)
    return convolve2d(img, kernel, **kwargs)


def gaussian_kernel(size=7, sigma=5):
    kernel = cv2.getGaussianKernel(size, sigma)
    return kernel @ kernel.T

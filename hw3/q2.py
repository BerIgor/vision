import cv2 as cv
import numpy as np


# input: image that was prepped for harris detection
def harris_detect(image):
    res = cv.cornerHarris(image, 5, 3, 0.04)
    return res


# input: image is an rgb image
def prep_image_for_harris(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray_image = np.float32(gray_image)
    return gray_image

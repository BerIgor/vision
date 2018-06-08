import cv2 as cv
import numpy as np


# input: image that was prepped for harris detection
def q2_harris_detect(image):
    res = cv.cornerHarris(image, 5, 3, 0.04)
    print(res)


# input: image is an rgb image
def q2_prep_image(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray_image = np.float32(gray_image)
    return gray_image

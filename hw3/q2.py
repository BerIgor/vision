import cv2 as cv
import numpy as np
from hw3 import utils


# input: image that was prepped for harris detection
def harris_detect(image):
    res = cv.cornerHarris(image, 5, 3, 0.04)
    return res


# input: image is an rgb image
def prep_image_for_harris(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray_image = np.float32(gray_image)
    return gray_image


def harris_and_nms(image, nms_window=30):
    """
    Extracts feature point from image using harris corner detector + non-maximum suppression
    :param image - image to find edges in
    :param nms_window - window size to perform non maximum suppression in
    :return: list of feature points (list of tuples of x,y coordinates)
    :return: The original image + the detected feature points (pixels) marked in red
    """

    # Find edges using Harris
    image_harris = harris_detect(prep_image_for_harris(image))
    # utils.cvshow("Harris", image_harris)

    # Set survivor nms points on image
    image_harris_nms = image.copy()
    harris_nms_mask, feature_points = utils.non_maximum_suppression(image_harris, nms_window)
    image_harris_nms[harris_nms_mask == np.max(harris_nms_mask)] = [0, 0, 255] # Window size of 64 was tested to return roughly 100 points on our frame
    # utils.cvshow("result", image_harris_nms)

    return  feature_points, image_harris_nms

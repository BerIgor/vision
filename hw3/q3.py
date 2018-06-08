import cv2 as cv
import numpy as np

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
colors = [red, green, blue]

max_points = 6


def mark_points(image, points):
    """
    Draws shapes on provided points. The shapes are six predefined shapes
    @:param image is the image to mark
    @:param points is a list of coordinates where to draw shapes
            each coordinate is a tuple (x, y)
    @:returns the image with the points marked using shapes
    """
    if len(points) > max_points:
        raise ValueError('More than 6 points in points')
    cv.circle(image, points[0], 5, red, -1)
    cv.circle(image, points[1], 5, green, -1)
    cv.circle(image, points[2], 5, blue, -1)

    top_left, bottom_right = get_rect(points[3])
    cv.rectangle(image, top_left, bottom_right, red, -1)
    top_left, bottom_right = get_rect(points[4])
    cv.rectangle(image, top_left, bottom_right, green, -1)
    top_left, bottom_right = get_rect(points[5])
    cv.rectangle(image, top_left, bottom_right, blue, -1)
    return image


def get_rect(point, edge=5):
    """
    Returns the rect defining the rectangle centered at point with edge size edge
    :param point: the center of the rectangle (square)
    :param edge: size of the desired edge of rectangle
    :return: point tuples for top left and bottom right corners of the rectangle
    """
    top_left = tuple(np.subtract(point, edge))
    bottom_right = tuple(np.add(point, edge))
    return top_left, bottom_right

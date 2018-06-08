import cv2 as cv


def cvshow(title, im):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, im)
    cv.waitKey()

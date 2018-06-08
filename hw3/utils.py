import cv2 as cv
import numpy as np


def cvshow(title, im):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, im)
    cv.waitKey()


def non_maximum_suppression(img, window_size):
    # slide a window across the image
    img_max = np.max(img)
    suppressed_img = np.zeros(img.shape)
    for y in range(0, img.shape[0], window_size):
        for x in range(0, img.shape[1], window_size):
            y_next = y + window_size if (y + window_size <= img.shape[0]) else img.shape[0]
            x_next = x + window_size if (x + window_size <= img.shape[1]) else img.shape[1]
            img_win = img[y:y_next, x:x_next]
            # cvshow("original win",img_win)
            win_max = np.max(img_win)
            ret, res = cv.threshold(img_win, 0.99*win_max, img_max, cv.THRESH_BINARY)
            # cvshow("nms win",res)
            suppressed_img[y:y_next, x:x_next] = res

    cvshow("After nms", suppressed_img)
    return suppressed_img



import cv2 as cv
import numpy as np


def cvshow(title, im):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, im)
    cv.waitKey()


def non_maximum_suppression(img, win_size):
    # slide a window across the image
    img_max = np.max(img)
    suppressed_img = np.zeros(img.shape)
    max_points_list = []
    for y in range(0, img.shape[0], win_size):
        for x in range(0, img.shape[1], win_size):
            # Extract current window
            y_next = y + win_size if (y + win_size <= img.shape[0]) else img.shape[0]
            x_next = x + win_size if (x + win_size <= img.shape[1]) else img.shape[1]
            img_win = img[y:y_next, x:x_next]
            # cvshow("original win",img_win)
            # NMS on window:
            win_max = np.max(img_win)
            for j in range(img_win.shape[0]):
                for k in range(img_win.shape[1]):
                    if (img_win[j, k] == win_max):
                        img_win[j, k] = img_max
                        max_points_list.append([x+j,y+k])
                    else:
                        img_win[j, k] = 0

            #ret, res = cv.threshold(img_win, 0.99*win_max, img_max, cv.THRESH_BINARY)
            # cvshow("nms win",res)
            suppressed_img[y:y_next, x:x_next] = img_win

    cvshow("After nms", suppressed_img)
    return suppressed_img, max_points_list


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


ref_feature_points = [(297, 187), (303, 236), (447, 92), (421, 309), (459, 360), (505, 154)]
frame1_feature_points = [(304, 148), (308, 199), (459, 38), (419, 256), (450, 299), (510, 116)]
frame2_feature_points = [(280, 225), (283, 263), (439, 90), (390, 302), (424, 357), (503, 166)]
frame3_feature_points = [(235, 234), (239, 297), (399, 97), (363, 341), (393, 387), (459, 193)]
frame4_feature_points = [(238, 233), (243, 268), (412, 71), (361, 314), (391, 358), (468, 179)]
frame5_feature_points = [(244, 239), (245, 292), (422, 85), (360, 318), (401, 382), (470, 209)]
frame6_feature_points = [(229, 265), (234, 299), (411, 74), (348, 298), (387, 364), (459, 209)]

frames_points = [ref_feature_points, frame1_feature_points, frame2_feature_points, frame3_feature_points,
                 frame4_feature_points, frame5_feature_points, frame6_feature_points]


def get_frames_points():
    return frames_points


def transform(coordinates, a, b):
    """

    :param coordinates: N coordinates in a matrix with shape 2xN
    :param a: the a transformation matrix
    :param b: the b transformation matrix
    :return: a 2xN array of the transformed coordinates
    """
    return np.dot(a, coordinates) + b


def cvshow(title, im):
    # cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.namedWindow(title)
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
            # NMS on window:
            win_max = np.max(img_win)
            for j in range(img_win.shape[0]):
                for k in range(img_win.shape[1]):
                    if (img_win[j, k] == win_max):
                        img_win[j, k] = img_max
                        max_points_list.append([x+j,y+k])
                    else:
                        img_win[j, k] = 0

            suppressed_img[y:y_next, x:x_next] = img_win

    return suppressed_img, max_points_list


def display_images(image_list, mode='RGB'):
    f, sps = plt.subplots(nrows=1, ncols=len(image_list))
    if mode == 'GRAY':
        plt.gray()

    for i in range(0, len(image_list)):
        if mode == 'RGB':
            image = np.flip(image_list[i], 2)
        else:
            image = image_list[i]
        sps[i].imshow(image)
        sps[i].axis('off')

    plt.show()
    return

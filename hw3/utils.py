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
    The coordinates used is (row, column)
    :param coordinates: N coordinates in a matrix with shape 2xN
    :param a: the a transformation matrix
    :param b: the b transformation matrix
    :return: a 2xN array of the transformed coordinates
    """
    return np.dot(a, np.vstack(coordinates)) + b


def cvshow(title, im):
    # cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.namedWindow(title)
    cv.imshow(title, im)
    cv.waitKey()


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

def check_out_of_range_points(point_list):
    # Used for debugging
    point_list_err = [point for point in point_list if point[0] > 720 or point[1] > 480]

    return point_list_err

def compare_two_images(img1, img2, title):
    merged = cv.hconcat((img1, img2))
    cvshow(title, merged)
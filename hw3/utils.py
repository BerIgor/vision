import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

ref_feature_points = [(297, 187), (303, 236), (447, 92), (421, 309), (459, 360), (505, 154)]
frame1_feature_points = [(304, 148), (308, 199), (459, 38), (419, 256), (450, 299), (510, 116)]
frame2_feature_points = [(280, 225), (283, 263), (439, 90), (390, 302), (424, 357), (503, 166)]
frame3_feature_points = [(235, 234), (239, 297), (399, 97), (363, 341), (393, 387), (459, 193)]
frame4_feature_points = [(238, 233), (243, 268), (412, 71), (361, 314), (391, 358), (468, 179)]
frame5_feature_points = [(244, 239), (245, 292), (422, 85), (360, 318), (401, 382), (470, 209)]
frame6_feature_points = [(229, 265), (234, 299), (411, 74), (348, 298), (387, 364), (459, 209)]

frames_points = [ref_feature_points, frame1_feature_points, frame2_feature_points, frame3_feature_points,
                 frame4_feature_points, frame5_feature_points, frame6_feature_points]

"""
The following two functions are here to convert our (x,y) lists to (r,c) lists
"""


def hstack_frames(frame_list, reverse=False):
    if reverse:
        merged = cv.hconcat(tuple(reversed(frame_list)))
    else:
        merged = cv.hconcat(tuple(frame_list))
    return merged


def rotate_frames(frame_list):
    rotated_frame_list = list()

    for frame in frame_list:
        rotated_frame_list.append(np.transpose(frame, (1, 0, 2)))
    return rotated_frame_list


def get_pwd():
    return os.getcwd().replace('\\', '//')


def invert_point_lists(point_lists):
    inverted_point_lists = list()
    for points in point_lists:
        inverted_point_lists.append(invert_points(points))
    return inverted_point_lists


def invert_points(points):
    inverted_points = list()
    for point in points:
        x, y = point
        new_point = (y, x)
        inverted_points.append(new_point)
    return inverted_points


"""
End of these stupid functions
"""


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


def video_save_frame(frame, main_dir, sub_dir, frame_number):
    path = str(main_dir) + '/our_data/' + str(sub_dir) + '/' + str(frame_number) + '.jpg'
    cv.imwrite(path, frame)
    return

def is_frame_good_pixel_count(frame):
    flat = np.sum(frame, 2)
    non_zeros = np.count_nonzero(flat)
    if non_zeros < 241920:
        return False
    return True


def is_frame_good_corner_check(frame, window_size=40, p=0.95):
    flat = np.sum(frame, 2)

    pixels_in_window = 2 * window_size
    min_non_zeros = int(p * pixels_in_window)

    non_zeros_clt = np.count_nonzero(flat[:window_size, :window_size])
    non_zeros_clb = np.count_nonzero(flat[-window_size:, :window_size])
    non_zeros_crt = np.count_nonzero(flat[:window_size, -window_size:])
    non_zeros_crb = np.count_nonzero(flat[-window_size:, -window_size])

    corners = [non_zeros_clt, non_zeros_clb, non_zeros_crt, non_zeros_crb]
    counter = 0
    for corner in corners:
        print(corner)
        if corner < min_non_zeros:
            counter += 1

    if counter >= 3:
        return False
    return True

def get_all_video_frames(video_path, rotate=False):
    video_reader = cv.VideoCapture(video_path)
    frames = list()
    more_frames = True
    while more_frames:
        more_frames, current_frame = video_reader.read()
        if more_frames is False:
            break

        if rotate:
            current_frame = np.transpose(current_frame, (1, 0, 2))

        frames.append(np.uint8(current_frame))
    return frames


def make_normal_video(output_video_path, frames):
    rows = frames[0].shape[0]
    cols = frames[0].shape[1]
    video_format = cv.VideoWriter_fourcc(*"XVID")
    video_writer = cv.VideoWriter(output_video_path, video_format, 30, (cols, rows))
    for frame in frames:
        video_writer.write(np.uint8(frame))
    video_writer.release()

def make_video_from_image_files(image_files_dir, rotate = False):
    images_path = get_pwd() + '/our_data/' + image_files_dir + '/'
    frame_list = [cv.imread(images_path + img) for img in os.listdir(images_path) if os.path.isfile(img)]
    if rotate:
        frame_list = [np.transpose(frame, (1, 0, 2)) for frame in frame_list]
    make_normal_video(images_path + 'vid.avi',frame_list)




"""
    total_pixels = np.size(frame)
    black_pixels = total_pixels - non_zeros
    if black_pixels > 207360:
        return False
    return True
"""

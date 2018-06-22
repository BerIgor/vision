import cv2
import numpy as np
import math


def get_pwd():
    return os.getcwd().replace('\\', '//')


def get_frames_uniform(video_path, number_of_frames):
    video_reader = cv2.VideoCapture(video_path)
    length = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    interval = math.floor(length/number_of_frames)
    frames = list()
    for i in range(number_of_frames):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i*interval)
        _, frame = video_reader.read()
        frames.append(np.uint8(frame))
    return frames


def get_all_frames(video_path):
    video_reader = cv2.VideoCapture(video_path)
    frames = list()
    more_frames = True
    while more_frames:
        more_frames, current_frame = video_reader.read()
        if more_frames is False:
            break
        frames.append(np.uint8(current_frame))
    return frames


def cvshow(title, im):
    # cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.namedWindow(title)
    cv.imshow(title, im)
    cv.waitKey()

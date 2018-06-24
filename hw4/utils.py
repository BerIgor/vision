import cv2
import numpy as np
import math
import os


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


def make_normal_video(output_video_path, frames):
    # input: output_video_path is the path where the resulting video is created
    # input: frames is a list containing frames
    # input: frame_duration is the duration in seconds each frame will be visible
    # input: fps is the desired fps. maybe if we use lower fps it'll be better for long videos
    # output: void
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    video_format = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_video_path, video_format, 30, (width, height))
    for frame in frames:
        video_writer.write(np.uint8(frame))
    video_writer.release()


def cvshow(title, im):
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.namedWindow(title)
    cv2.imshow(title, im)
    cv2.waitKey()

def xy_vec_to_tuples_list(xy_vec):
    return [(xy_vec[i], xy_vec[i + 1]) for i in range(0, xy_vec.size, 2)]

def get_sub_image(image, point, w_size):
    min_x = max(point[0] - w_size, 0)
    min_y = max(point[1] - w_size, 0)
    max_x = min(point[0] + w_size, image.shape[1])
    max_y = min(point[1] + w_size, image.shape[0])

    sub_image = image[min_y:max_y + 1, min_x:max_x + 1]
    return sub_image

def video_save_frame(frame, main_dir, sub_dir, frame_number):
    path = str(main_dir) + '/our_data/' + str(sub_dir) + '/' + str(frame_number) + '.jpg'
    cv2.imwrite(path, frame)
    return

def clean_output_directories(images_dir_path, stab_images_dir_path=None):
    import shutil

    # Delete and re-create current output folders
    if os.path.isdir(images_dir_path):
        shutil.rmtree(images_dir_path)
    try:
        original_umask = os.umask(0)
        os.makedirs(images_dir_path)
    finally:
        os.umask(original_umask)

    if stab_images_dir_path is not None:
        if os.path.isdir(stab_images_dir_path):
            shutil.rmtree(stab_images_dir_path)
        try:
            original_umask = os.umask(0)
            os.makedirs(stab_images_dir_path)
        finally:
            os.umask(original_umask)


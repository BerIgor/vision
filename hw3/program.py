import sys, os
import cv2 as cv
import math
import numpy as np

from hw3 import *

# Globals
pwd = os.getcwd().replace('\\','//')


def get_frames_uniform(video_path, number_of_frames):
    video_reader = cv.VideoCapture(video_path)
    length = video_reader.get(cv.CAP_PROP_FRAME_COUNT)
    interval = math.floor(length/(number_of_frames+1))
    frames = list()
    for i in range(number_of_frames+1):
        video_reader.set(cv.CAP_PROP_POS_FRAMES, i*interval)
        _, frame = video_reader.read()
        frames.append(frame)
    return frames


# input: output_video_path is the path where the resulting video is created
# input: frames is a list containing frames
# input: frame_duration is the duration in seconds each frame will be visible
# input: fps is the desired fps. maybe if we use lower fps it'll be better for long videos
# output: void
def q1_make_video(output_video_path, frames, frame_duration, fps=30):
    number_of_frames = fps*frame_duration

    rows = frames[0].shape[0]
    cols = frames[0].shape[1]
    video_format = cv.VideoWriter_fourcc(*"XVID")
    video_writer = cv.VideoWriter(output_video_path, video_format, fps, (cols, rows)) # In the constructor (column, row). However in video_writer.write its (row, column).
    for frame in frames:
        for i in range(number_of_frames):
            video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    print("Welcome to hw3")
    source_video_path = pwd + '/our_data/ariel.mp4'
    frame_list = get_frames_uniform(source_video_path, 6)

    image = frame_list[0]
    image = q2.prep_image_for_harris(image)
    res = q2.harris_detect(image)
    utils.cvshow("Test", res)

    # result_video_path = pwd + '/our_data/result.avi'
    # q1_make_video(result_video_path, frame_list, 3)

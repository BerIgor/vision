import sys, os
import cv2 as cv
import math
import numpy as np
from hw3 import *


# Globals
pwd = os.getcwd().replace('\\','//')


if __name__ == "__main__":
    # close all open windows
    cv.destroyAllWindows()
    print("Welcome to hw3")

    source_video_path = pwd + '/our_data/ariel.mp4'
    frame_list = utils.get_frames_uniform(source_video_path, 7, rotate=False)

    # Q2
    q2.perform(frame_list)

    # Q3
    q3_frame_list = utils.get_frames_uniform(source_video_path, 7, rotate=False)
    q3.perform(q3_frame_list)

    # Q4
    q4.perform()
    q4_transformations = q4.get_seq_transformation(utils.get_frames_points())

    # Q5
    q5.perform(frame_list, q4_transformations)
    q5_stabilized_frames = q5.stabilize_frames(frame_list, q4_transformations)
    utils.q1_make_video(pwd + '/q5_ariel_stable.avi', q5_stabilized_frames, 1)

    # Q6
    q6.answer_question(frame_list)

    # Q7

    # Q8
    stab_short = q8.perform(frame_list, nms_window=35, search_window=5, ssd_window=20)
    stab_short = utils.rotate_frames(stab_short)
    new_stab_list = list()
    for image in stab_short:
        image = np.flip(image, 1)
        new_stab_list.append(image)
    utils.q1_make_video(utils.get_pwd() + '/q8_short_stable.avi', new_stab_list, 1)

    stab_short = utils.hstack_frames(new_stab_list, reverse=False)
    cv.imwrite(utils.get_pwd() + '/q8_short_stable.jpg', stab_short)

    # make full video
    full_frame_list = utils.get_all_video_frames(source_video_path, rotate=False)
    stab_long = q8.perform(full_frame_list, nms_window=35, search_window=5, ssd_window=20)
    stab_long = utils.rotate_frames(stab_long)
    i = 0
    frame_list_flipped = list()
    for frame in stab_long:
        frame = np.flip(frame, 1)
        frame_list_flipped.append(frame)
        utils.video_save_frame(frame, utils.get_pwd(), 'stab_8', i)
        i += 1
    utils.make_normal_video(utils.get_pwd() + '/q8_long_stable.avi', frame_list_flipped)

    # Q9
    frame_list = utils.get_all_video_frames(source_video_path)
    output_video_path = pwd + '/our_data/ariel_stabilized_q9.avi'
    frame_list_stabilized = q9.perform_subspace_video_stabilization(frame_list, output_video_path)

    # END


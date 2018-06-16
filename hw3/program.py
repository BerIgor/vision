import sys, os
import cv2 as cv
import math
import numpy as np
import random
from hw3 import *

# Globals
pwd = os.getcwd().replace('\\','//')


def get_all_video_frames(video_path):
    video_reader = cv.VideoCapture(video_path)
    frames = list()
    more_frames = True
    while more_frames:
        more_frames, current_frame = video_reader.read()
        if more_frames is False:
            break
        frames.append(current_frame)

    return frames


def get_frames_uniform(video_path, number_of_frames):
    video_reader = cv.VideoCapture(video_path)
    length = video_reader.get(cv.CAP_PROP_FRAME_COUNT)
    interval = math.floor(length/number_of_frames)
    frames = list()
    for i in range(number_of_frames):
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
            video_writer.write(np.uint8(frame))
    video_writer.release()


def make_normal_video(output_video_path, frames):

    rows = frames[0].shape[0]
    cols = frames[0].shape[1]
    video_format = cv.VideoWriter_fourcc(*"XVID")
    video_writer = cv.VideoWriter(output_video_path, video_format, 30, (cols, rows))
    for frame in frames:
        video_writer.write(np.uint8(frame))
    video_writer.release()


if __name__ == "__main__":
    # close all open windows
    cv.destroyAllWindows()
    print("Welcome to hw3")
    source_video_path = pwd + '/our_data/ariel.mp4'
    frame_list = get_frames_uniform(source_video_path, 7)

    """
    points_to_mark = utils.get_frames_points()
    for i in range(len(frame_list)):
        frame = frame_list[i]
        points = points_to_mark[i]
        marked_image = q3.mark_points(frame, points)
        utils.cvshow("FRAME", marked_image)
    """
    """
    xy_point_lists = utils.get_frames_points()
    rc_point_lists = utils.invert_point_lists(xy_point_lists)

    transformation_list = q4.get_seq_transformation(rc_point_lists)
    stabilized_images = q5.perform(frame_list, transformation_list)
    stabilized_video_path = pwd + '/our_data/q5_ariel_stable.avi'
    q1_make_video(stabilized_video_path, stabilized_images, 2)
    exit()

    """
    frame_list_q8 = get_all_video_frames(source_video_path)
    """
    frames_point_pairs = list()
    transformations = list()
    
    for frame in frame_list_q8:

        ref_points, seq_points = q6.perform_q6(frame_list_q8[0], frame)
        # ref_points = utils.invert_points(ref_points)
        # seq_points = utils.invert_points(seq_points)
        transformations.append(q7.calc_transform_ransac(ref_points, seq_points))
    """
    stabilized_frames_q8 = q8.perform(frame_list_q8[:10])

    stabilized_video_path = pwd + '/our_data/q8_ariel_stable.avi'
    make_normal_video(stabilized_video_path, stabilized_frames_q8)

    exit()

    # Find edges using Harris
    # image_harris_nms = q2.harris_and_nms(image)
    # plist = [(20,20), (90,90), (300,300), (300,400), (50, 270), (270,400), (350, 350)]
    # q3.mark_points(image, plist)
    # utils.cvshow("marked", q3.mark_points(image, plist))

    # Test q3 - Choose manually matching feature points
    '''
    q3.choose_match_points_for_all_frames(frame_list)

    # Get manually matched points
    frames_feature_points_list = utils.get_frames_points()

    # Test manual points:
    ref = frame_list[0]
    for i in range(len(frame_list[1:])):
        frame = frame_list[i+1]
        frame_feature_points = frames_feature_points_list[i]
        final_ref_with_matkings = q3.mark_points(ref, ref_feature_points)
        final_frame_with_markings = q3.mark_points(frame, frame_feature_points)
        finals_merged = cv.hconcat((final_ref_with_matkings,final_frame_with_markings))
        utils.cvshow("final results - ref vs frame " + str(i+1), finals_merged)
    '''

    # Test q4 - finding affine transformation
    '''
    points = list()
    # points.append([(1, 1), (2, 2), (3, 3)])
    # points.append([(2, 2), (3, 3), (4, 4)])
    points.append(ref_feature_points)
    points.append(frame1_feature_points)

    res_list = q4.get_seq_transformation(points)
    a, b = res_list[0]

    for ref_point in ref_feature_points[3:]:
        print("Reference Point")
        print(ref_point)
        points_transformed = q4.test_transformation2(np.vstack(ref_point), a, b)
        print("Transformed points:")
        print((tuple(np.squeeze(points_transformed))))
    '''

    # Test q5 - Stabillization
    '''
    ret, res = cv.threshold(res, 0.01 * res.max(), 255, cv.THRESH_BINARY)

    result_video_path = pwd + '/our_data/result.avi'
    q1_make_video(result_video_path, frame_list, 3)
    '''

    # Test q6
    '''
    from random import shuffle

    ref = frame_list[0].copy()
    test_points_num = 10
    for frame in frame_list[1:]:
        match_frame = frame.copy()
        ref_feature_points, matched_points = q6.perform_q6(ref, match_frame)


        print("ref points num: " + str(len(ref_feature_points)) + "\nframe points num: " + str(len(matched_points)))
        indices = list(range(len(ref_feature_points)))
        shuffle(indices)
        point_indices = indices[0:test_points_num]
        print(type(point_indices[0]))
        ref_feature_points_test = [ref_feature_points[ind] for ind in point_indices]
        match_frame_points_test = [matched_points[ind] for ind in point_indices]
        print("ref points test num: " + str(len(ref_feature_points_test)) + "\nframe points test num: " + str(len(match_frame_points_test)))
        # final_ref_with_matkings = np.transpose(q3.mark_points(ref, ref_feature_points_test), (1, 0, 2))
        # final_frame_with_markings = np.transpose(q3.mark_points(match_frame, match_frame_points_test), (1, 0, 2))
        # finals_merged = cv.hconcat((final_ref_with_matkings,final_frame_with_markings))
        for ip in range(len(ref_feature_points_test)):
            print("Ref p: " + str(ref_feature_points_test[ip]) + " Match p: " + str(match_frame_points_test[ip]))
        final_ref_with_markings = q3.mark_points(ref, ref_feature_points_test)
        final_frame_with_markings = q3.mark_points(match_frame, match_frame_points_test)
        utils.compare_two_images(final_ref_with_markings, final_frame_with_markings, "harris and nms - ref vs frame")
        ref = frame_list[0].copy() # Reset ref image
    '''




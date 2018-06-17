import sys, os
import cv2 as cv
import math
import numpy as np
import random
from hw3 import *
from hw3 import igor_playground


# Globals
pwd = os.getcwd().replace('\\','//')


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


def get_frames_uniform(video_path, number_of_frames, rotate=False):
    video_reader = cv.VideoCapture(video_path)
    length = video_reader.get(cv.CAP_PROP_FRAME_COUNT)
    interval = math.floor(length/number_of_frames)
    frames = list()
    for i in range(number_of_frames):
        video_reader.set(cv.CAP_PROP_POS_FRAMES, i*interval)
        _, frame = video_reader.read()
        if rotate:
            frame = np.transpose(frame, (1, 0, 2))
        frames.append(np.uint8(frame))
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
    frame_list = get_frames_uniform(source_video_path, 7, rotate=False)

    # Q3
    q3_frame_list = get_frames_uniform(source_video_path, 7, rotate=False)
    q3.perform(q3_frame_list)

    # Q4
    q4_transformations = q4.get_seq_transformation(utils.get_frames_points())

    # Q5
    q5_stabilized_frames = q5.perform(frame_list, q4_transformations)
    """
    frames_points = utils.get_frames_points()
    for i in range(len(q5_stabilized_frames)):
        q3.mark_points(q5_stabilized_frames[i], frames_points[i])
        utils.cvshow("stab2", q5_stabilized_frames[i])
    """
    make_normal_video(pwd + '/our_data/q5_ariel_stable.avi', q5_stabilized_frames)

    # Q6
    mask = cv.imread(pwd + '/our_data/masked_frames/0.jpg')
    mask = np.transpose(mask, (1, 0, 2))
    print(len(frame_list))
    for frame in frame_list:
        print("frame")
        ref_feature_points, matched_points = q6.perform_q6(frame_list[0], frame, mask)
    # TODO: Complete

    # Q8
    q8_all_frame_list = get_all_video_frames(source_video_path)

    q8_trans_list = list()
    i = 0
    for frame in q8_all_frame_list:
        ref_feature_points, matched_points = q6.perform_q6(q8_all_frame_list[0], frame, mask)
        q8_transformation = q7.calc_transform_ransac(ref_feature_points, matched_points)
        q8_trans_list.append(q8_transformation)
        print(len(q8_trans_list))
        a, b = q8_trans_list[i]
        stab_image = q5.stabilize_image_cv(frame, a, b)
        utils.video_save_frame(stab_image, pwd, 'stab_8', i)
        i += 1

    print(len(q8_all_frame_list))
    print(len(q8_trans_list))

    q8_stab_list = list()
    for i in range(len(q8_all_frame_list)):
        a, b = q8_trans_list[i]
        stab_image = q5.stabilize_image_cv(q8_all_frame_list[i], a, b)
        utils.video_save_frame(stab_image, pwd, "stab_8", i)
        q8_stab_list.append(stab_image)

    make_normal_video(pwd + '/our_data/q8_ariel_stable.avi', q8_stab_list)
    exit()

    frame_list = get_all_video_frames(source_video_path, rotate=True)
    # masked_frame_list = get_all_video_frames(str(pwd) + '/our_data/ariel_segmented_full_length.avi')
    print(np.shape(frame_list[0]))
    igor_playground.stabilize_using_mask(frame_list, pwd)
    exit()
    """
    points_to_mark = utils.get_frames_points()
    for i in range(len(frame_list)):
        frame = frame_list[i]
        points = points_to_mark[i]
        marked_image = q3.mark_points(frame, points)
        utils.cvshow("FRAME", marked_image)


    xy_point_lists = utils.get_frames_points()
    rc_point_lists = utils.invert_point_lists(xy_point_lists)

    transformation_list = q4.get_seq_transformation(rc_point_lists)
    stabilized_images = q5.perform(frame_list, transformation_list)
    stabilized_video_path = pwd + '/our_data/q5_ariel_stable.avi'
    q1_make_video(stabilized_video_path, stabilized_images, 2)
    exit()


    frame_list_q8 = get_all_video_frames(source_video_path)

    frames_point_pairs = list()
    transformations = list()
    
    for frame in frame_list_q8:

        ref_points, seq_points = q6.perform_q6(frame_list_q8[0], frame)
        # ref_points = utils.invert_points(ref_points)
        # seq_points = utils.invert_points(seq_points)
        transformations.append(q7.calc_transform_ransac(ref_points, seq_points))

    stabilized_frames_q8 = q8.perform(frame_list_q8)

    i = 0
    for frame in frame_list:
        utils.video_save_frame(frame, "orig", i)
        i += 1

    i = 0
    for frame in stabilized_frames_q8:
        utils.video_save_frame(frame, pwd, "stab", i)
        i += 1

    stabilized_video_path = pwd + '/our_data/q8_milk_stable.avi'
    make_normal_video(stabilized_video_path, stabilized_frames_q8)

    exit()

    # Find edges using Harris
    # image_harris_nms = q2.harris_and_nms(image)
    # plist = [(20,20), (90,90), (300,300), (300,400), (50, 270), (270,400), (350, 350)]
    # q3.mark_points(image, plist)
    # utils.cvshow("marked", q3.mark_points(image, plist))

    # Test q3 - Choose manually matching feature points


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


    # Test q4 - finding affine transformation

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


    # Test q5 - Stabillization

    ret, res = cv.threshold(res, 0.01 * res.max(), 255, cv.THRESH_BINARY)

    result_video_path = pwd + '/our_data/result.avi'
    q1_make_video(result_video_path, frame_list, 3)

    """
    # Test q6

    # ref = cv.imread(str(pwd) + '/our_data/orig/0.jpg')
    # milk = cv.imread(str(pwd) + '/our_data/stab/14.jpg')
    ref = frame_list[0]
    i = 1
    for frame in frame_list[1:]:
        milk = frame_list[14]

        print(np.shape(ref))
        print(np.shape(milk))
        from random import shuffle

        ref_feature_points, matched_points = q6.perform_q6(ref, milk)

        ref_mark = q3.mark_points(ref, ref_feature_points)
        milk_mark = q3.mark_points(milk, matched_points)

        print(ref_mark.shape)
        print(milk_mark.shape)
        # utils.cvshow("REF", ref_mark)
        # utils.cvshow("14", milk_mark)

        a, b = q7.calc_transform_ransac(ref_feature_points, matched_points)
        stab_milk = q5.stabilize_image_cv(milk, a, b)
        # utils.cvshow("STAB", stab_milk)
        path = str(pwd) + '/our_data/stab_new/' + str(i) + '.jpg'
        print(np.shape(ref_mark))
        print(np.shape(stab_milk))
        merged = cv.hconcat((ref_mark, stab_milk))
        cv.imwrite(path, merged)

        i += 1
    """
    # for frame in frame_list[14]:
    frame = milk

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


    # Test q9
    all_video_frames = get_all_video_frames(source_video_path)
    q9.perform_subspace_video_stabilization(all_video_frames)
    """
    # Igor testing for q9 start here

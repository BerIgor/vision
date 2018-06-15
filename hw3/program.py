import sys, os
import cv2 as cv
import math
import numpy as np
import random
from hw3 import *

# Globals
pwd = os.getcwd().replace('\\','//')


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


if __name__ == "__main__":
    # close all open windows
    cv.destroyAllWindows()
    print("Welcome to hw3")
    source_video_path = pwd + '/our_data/ariel.mp4'
    frame_list = get_frames_uniform(source_video_path, 7)

    arr = np.array([[1, 2], [3, 4]])
    print(arr[1, 0])

    """
    points_to_mark = utils.get_frames_points()
    for i in range(len(frame_list)):
        frame = frame_list[i]
        points = points_to_mark[i]
        marked_image = q3.mark_points(frame, points)
        utils.cvshow("FRAME", marked_image)
    """
    stabilized_frames_q8 = q8.perform(frame_list, utils.get_frames_points())
    stabilized_video_path = pwd + '/our_data/q5_ariel_stable.avi'
    q1_make_video(stabilized_video_path, stabilized_frames_q8, 2)
    exit()
    # q3.perform(frame_list)
    transformation_list = q4.get_seq_transformation(utils.get_frames_points())
    stabilized_images = q5.perform(frame_list, transformation_list)
    print(len(stabilized_images))
    stabilized_video_path = pwd + '/our_data/q5_ariel_stable.avi'
    q1_make_video(stabilized_video_path, stabilized_images, 2)


    # Find edges using Harris
    # image_harris_nms = q2.harris_and_nms(image)
    # plist = [(20,20), (90,90), (300,300), (300,400), (50, 270), (270,400), (350, 350)]
    # q3.mark_points(image, plist)
    # utils.cvshow("marked", q3.mark_points(image, plist))

    # Choose matching feature points
    # q3.choose_match_points_for_all_frames(frame_list)

    # Matching of feature points, chosen manually:
    # ref_feature_points = [(297, 187), (303, 236), (447, 92), (421, 309), (459, 360), (505, 154)]
    # frame1_feature_points = [(304, 148), (308, 199), (459, 38), (419, 256), (450, 299), (510, 116)]
    # frame2_feature_points = [(280, 225), (283, 263), (439, 90), (390, 302), (424, 357), (503, 166)]
    # frame3_feature_points = [(235, 234), (239, 297), (399, 97), (363, 341), (393, 387), (459, 193)]
    # frame4_feature_points = [(238, 233), (243, 268), (412, 71), (361, 314), (391, 358), (468, 179)]
    # frame5_feature_points = [(244, 239), (245, 292), (422, 85), (360, 318), (401, 382), (470, 209)]
    # frame6_feature_points = [(229, 265), (234, 299), (411, 74), (348, 298), (387, 364), (459, 209)]
    #
    # # Test q4 - finding affine transformation
    # points = list()
    # # points.append([(1, 1), (2, 2), (3, 3)])
    # # points.append([(2, 2), (3, 3), (4, 4)])
    # points.append(ref_feature_points)
    # points.append(frame1_feature_points)
    #
    # res_list = q4.get_seq_transformation(points)
    # a, b = res_list[0]
    #
    # for ref_point in ref_feature_points[3:]:
    #     print("Reference Point")
    #     print(ref_point)
    #     points_transformed = q4.test_transformation2(np.vstack(ref_point), a, b)
    #     print("Transformed points:")
    #     print((tuple(np.squeeze(points_transformed))))

    # Test q5 - Stabillization
    # ret, res = cv.threshold(res, 0.01 * res.max(), 255, cv.THRESH_BINARY)

    # result_video_path = pwd + '/our_data/result.avi'
    # q1_make_video(result_video_path, frame_list, 3)

    # Test q6
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
        final_ref_with_matkings = q3.mark_points(ref, ref_feature_points_test)
        final_frame_with_markings = q3.mark_points(match_frame, match_frame_points_test)
        finals_merged = cv.hconcat((final_ref_with_matkings,final_frame_with_markings))
        utils.cvshow("final results - ref vs frame", finals_merged)
        ref = frame_list[0].copy() # Reset ref image

    # utils.cvshow("Ref image feature", q3.mark_points(ref, ref_feature_points_test))
    # utils.cvshow("Frame - Auto detected matching features", q3.mark_points(match_frame, match_frame_points_test))




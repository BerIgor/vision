import cv2
import numpy as np
from hw3 import *


def stabilize_using_mask(frame_list, pwd):
    stabilized_frame_list = list()

    ref_frame = frame_list[0]
    stabilized_frame_list.append(ref_frame)
    mask = cv2.imread(pwd + '/our_data/masked_frames/0.jpg')

    i = 0
    for seq_frame in frame_list[1:]:
        ref_feature_points, matched_points = q6.perform_q6(ref_frame, seq_frame, mask)
        a, b = q7.calc_transform_ransac(ref_feature_points, matched_points)

        stab = q5.stabilize_image_cv(seq_frame, a, b)
        cv2.imwrite(str(pwd) + '/our_data/stab_new/' + str(i) + '.jpg', stab)
        i += 1


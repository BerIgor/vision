import numpy as np
from hw3 import q5
from hw3 import q6
from hw3 import q7
from hw3 import utils
import cv2 as cv


def perform(frames, search_window, ssd_window, nms_window):
    mask = cv.imread(utils.get_pwd() + '/our_data/masked_frames/0.jpg')
    mask = np.transpose(mask, (1, 0, 2))
    
    transformations = list()
    i = 0
    for frame in frames:
        print("frame " + str(i))
        ref_points, seq_points = q6.perform_q6(frames[0], frame, mask,
                                               search_win=search_window,
                                               ssd_win=ssd_window,
                                               nms_window=nms_window)

        if len(ref_points) < 3:
            ref_points, seq_points = q6.perform_q6(frames[0], frame, mask,
                                                   search_win=2*search_window,
                                                   ssd_win=ssd_window,
                                                   nms_window=nms_window)
        transformations.append(q7.calc_transform_ransac(ref_points, seq_points))
        i += 1

    stabilized_images = q5.stabilize_frames(frames, transformations)
    return stabilized_images

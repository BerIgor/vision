import numpy as np
from hw3 import q5
from hw3 import q6
from hw3 import q7


def perform(frames):
    transformations = list()
    for frame in frames:
        ref_points, seq_points = q6.perform_q6(frames[0], frame)
        transformations.append(q7.calc_transform_ransac(ref_points, seq_points))

    stabilized_images = q5.perform(frames, transformations)
    return stabilized_images

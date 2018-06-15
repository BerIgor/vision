import numpy as np

from hw3 import q5
from hw3 import q7


def perform(frames, frames_points):
    transformations = list()
    a = np.array([[1, 0], [0, 1]])
    b = np.array([[0], [0]])
    transformations.append((a, b))
    ref_points = frames_points[0]
    for seq_points in frames_points[1:]:
        transformations.append(q7.calc_transform_ransac(ref_points, seq_points))
    stabilized_images = q5.perform(frames, transformations)
    return stabilized_images

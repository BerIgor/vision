import numpy as np
from hw3 import q7


def calculate_transformations(ref_window_points, seq_window_points):
    transformations = list()



def calculate_transformation(ref_points, seq_points):
    """

    :param ref_points: is column vector of [[x1], [y1], [x2]...] which represent the reference points
    :param seq_points: is column vector of [[x1], [y1], [x2]...] which represent the transformed points
    :return: returns the (a, b) transformation matrices
    """

    ref_points = np.reshape(ref_points, (1, -1))
    seq_points = np.reshape(seq_points, (1, -1))
    return q7.calc_transform_ransac(ref_points, seq_points)
import numpy as np


def get_transformation(points_reference, points_transformed):
    """
    Calculates the affine transformation between the points using least squares
    Input example: [(1, 1), (2, 2), (3, 3)]
    :param points_reference: is a list of tuples representing the original points
    :param points_transformed: is a list of tuples representing the points after transformation
    :return: the transformation matrix
    """
    x = np.array(points_reference)
    y_ = np.array(points_transformed)
    y = np.insert(y_, 2, 1, axis=1)
    parameters, _, a_rank, _ = np.linalg.lstsq(x, y)
    a = parameters[:, [0, 1]]
    b = parameters[:, [2]]
    return a, b


def get_seq_transformation(points):
    """
    Calculates the transformation from the first point list (points[0]) to every subsequent point list
    :param points: a list of lists, each inner list consists of point tuples
    :return: a list of transformation matrices, each is a tuple
    """
    transformations = list()
    reference = points[0]
    for frame_points in points[1:]:
        current_transformation = get_transformation(reference, frame_points)
        transformations.append(current_transformation)
    return transformations


def test_transformation():
    points_reference = [(1, 1), (2, 2), (3, 3)]
    points_transformed = [(2, 2), (3, 3), (4, 4)]
    # example thing
    # points_reference = [[1, 1], [2, 2], [3, 3]]
    # points_transformed = [[2, 2, 1], [3, 3, 1], [4, 4, 1]]
    a, b = get_transformation(points_reference, points_transformed)
    print(np.dot(a, np.array([[3], [3]])) + b)

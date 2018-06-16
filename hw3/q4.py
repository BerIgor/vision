import numpy as np


def get_transformation2(points_reference, points_transformed):
    x = np.matrix(points_reference)
    y = np.matrix(points_transformed)

    # add ones on the bottom of x and y
    x = np.hstack((x,[1,1,1,1,1,1]))
    y = np.hstack((y,[1,1,1,1,1,1]))
    print(x)
    print(y)
    exit()
    print(parameters)
    # solve for A2
    parameters, _, a_rank, _ = np.linalg.lstsq(x, y)

    exit()
    # A2 = y * x.I
    # return function that takes input x and transforms it
    # don't need to return the 4th row as it is
    return lambda x: (A2*np.vstack((np.matrix(x).reshape(3,1),1)))[0:3,:]


def get_transformation(points_reference, points_transformed):
    print("In transformation")
    print(points_reference)
    print(points_transformed)
    print("Exit transformation")
    """
    Calculates the affine transformation between the points using least squares
    Input example: [(1, 1), (2, 2), (3, 3)]
    :param points_reference: is a list of tuples representing the original points
    :param points_transformed: is a list of tuples representing the points after transformation
    :return: the transformation matrix
    """
    x = np.array(points_reference)
    v = np.reshape(np.ones(x.shape[0]), (-1, 1))
    x = np.hstack((x, v))
    y = (np.array(points_transformed)).transpose()
    y = (np.vstack((y, np.ones(y.shape[1])))).transpose()

    parameters, _, a_rank, _ = np.linalg.lstsq(x, y)
    parameters[np.abs(parameters) < 1e-10] = 0
    a = parameters[0:2, 0:2]
    b = parameters[0:2, 2]
    b = np.reshape(b, (2, 1))
    print(parameters)
    print(a)
    print(b)
    return a, b


def get_seq_transformation(points):
    """
    Calculates the transformation from the first point list (points[0]) to every subsequent point list
    :param points: a list of lists, each inner list consists of point tuples
    :return: a list of transformation matrices, each is a tuple (a,b), including the id transform from ref to ref
    """
    transformations = list()
    id_transformation = (np.diag([1, 1]), np.array([[0], [0]]))
    reference = points[0]
    transformations.append(id_transformation)
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


def test_transformation2(points_reference, a, b):
    # print(a)
    # print(points_reference)
    # print(b)
    return np.dot(a, points_reference) + b


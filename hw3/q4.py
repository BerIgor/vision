import numpy as np



def get_transformation(points_reference, points_transformed):
    """
    Calculates the affine transformation between the points using least squares
    :param points_reference: is a list of tuples representing the original points
    :param points_transformed: is a list of tuples representing the points after transformation
    :return:
    """
    # points_reference, points_transformed
    points_reference = [[1, 1], [2, 2], [3, 3]]
    points_transformed = [[2, 2, 1], [3, 3, 1], [4, 4, 1]]
    x = np.array(points_reference)
    y = np.array(points_transformed)

    parameters, _, a_rank, _ = np.linalg.lstsq(x, y)
    a = parameters[:, [0, 1]]
    b = parameters[:, [2]]

    print(a)
    print(b)
    print(np.dot(a, np.array([[3], [3]])) + b)

    #print(np.array(parameters))
    #a = np.delete(parameters, 2, 1)


def test_transformation():
    points_reference = [(1, 1), (2, 2), (3, 3)]
    points_transformed = [(2, 2, 1), (3, 3, 1), (4, 4, 1)]
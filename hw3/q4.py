import numpy as np
from hw3 import utils


def perform():
    transformations = get_seq_transformation(utils.get_frames_points())
    for trans in transformations:
        a, b = trans

        low_values_flags = a < 0.005
        a[low_values_flags] = 0

        low_values_flags = b < 0.005
        b[low_values_flags] = 0

        m = np.hstack((a, b))
        print("===========================")
        np.set_printoptions(precision=2)
        print(m.astype(np.float))


def get_transformation(points_reference, points_transformed):
    """
    THE POINTS MUST BE (X, Y) for row and column
    Calculates the affine transformation between the points using least squares
    Input example: [(1, 1), (2, 2), (3, 3)]
    :param points_reference: is a list of tuples representing the original points
    :param points_transformed: is a list of tuples representing the points after transformation
    :return: the transformation matrix
    """

    x = np.array(points_transformed)

    v = np.reshape(np.ones(x.shape[0]), (-1, 1))
    x = np.hstack((x, v))
    y = np.array(points_reference)
    y = np.hstack((y, v))

    parameters, _, a_rank, _ = np.linalg.lstsq(x, y)

    # parameters[np.abs(parameters) < 1e-10] = 0
    a = np.transpose(parameters[0:2, 0:2])
    b = parameters[2, 0:2]
    b = np.reshape(b, (-1, 1))

    return a, b


def get_seq_transformation(points):
    """
    Calculates the transformation from the first point list (points[0]) to every subsequent point list
    :param points: a list of lists, each inner list consists of point tuples
    :return: a list of transformation matrices, each is a tuple (a,b), including the id transform from ref to ref
    """
    transformations = list()

    for frame_points in points:
        current_transformation = get_transformation(points[0], frame_points)
        transformations.append(current_transformation)
    return transformations


def test_transformation():
    points_ref = [(1, 1), (2, 2), (3, 3)]
    points_seq = [(2, 2), (4, 4), (6, 6)]
    a, b = get_transformation(points_seq, points_ref)
    print(a)
    print(b)
    print(np.dot(a, np.array([[3], [3]])) + b)


def test_transformation2():
    primary = np.array([[40., 1160., 0.],
                        [40., 40., 0.],
                        [260., 40., 0.],
                        [260., 1160., 0.]])

    secondary = np.array([[610., 560., 0.],
                          [610., -560., 0.],
                          [390., -560., 0.],
                          [390., 560., 0.]])

    # Pad the data with ones, so that our transformation can do translations too

    x = np.hstack([primary, np.ones((primary.shape[0], 1))])
    y = np.hstack([secondary, np.ones((secondary.shape[0], 1))])

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    a, res, rank, s = np.linalg.lstsq(x, y)
    print(a)
    # transform = lambda x: unpad(np.dot(pad(x), A))


def test_transformation3():
    """
    [[(297, 187), (303, 236), (447, 92), (421, 309), (459, 360), (505, 154)],
     [(304, 148), (308, 199), (459, 38), (419, 256), (450, 299), (510, 116)],
     [(280, 225), (283, 263), (439, 90), (390, 302), (424, 357), (503, 166)]]
     """
    primary = np.array([[297., 187.],
                        [303., 236.],
                        [447., 92.],
                        [421., 309.]])

    secondary = np.array([[297., 187.],
                         [303., 236.],
                         [447., 92.],
                         [421., 309.]])

    secondary2 = np.array([[304., 560.],
                          [308., 199.],
                          [459., 38.],
                          [419., 256.]])

    # Pad the data with ones, so that our transformation can do translations too

    x = np.hstack([primary, np.ones((primary.shape[0], 1))])
    print('hello')
    print(x)
    y = np.hstack([secondary, np.ones((secondary.shape[0], 1))])
    print('hello')
    print(y)
    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    a, res, rank, s = np.linalg.lstsq(x, y)
    print(a)


if __name__ == "__main__":
    test_transformation3()

    y = [(297, 187), (303, 236), (447, 92)]
    x = [(297, 187), (303, 236), (447, 92)]
    a, b = get_transformation(y, x)
    print(a)
    print(b)
    # x = [(304, 148), (308, 199), (459, 38)]
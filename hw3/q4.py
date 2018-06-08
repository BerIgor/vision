import numpy as np


def get_transformation():
    # points_reference, points_transformed
    points_reference = [[1, 1], [2, 2], [3, 3]]
    points_transformed = [[2, 2, 1], [3, 3, 1], [4, 4, 1]]
    x = np.transpose(np.array(points_reference))
    y = np.transpose(np.array(points_transformed))
    print(x)
    print(y)
    a, b, a_rank, d = np.linalg.lstsq(x, y)
    print(a)
    print(b)
    print(a_rank)
    print(d)
    print(np.multiply(a, np.matrix.transpose(x)))



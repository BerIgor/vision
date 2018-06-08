import numpy as np


def stabilize_image(image, a, b):
    """

    :param image: is the image to stabilize
    :param a: is the a matrix of an affine transformation
    :param b: is the b matrix of an affine transformation
    :return: a stabilized image
    """
    stabilized_image = np.zeros(np.shape(image))

    for x in range(stabilized_image.shape[0]):
        for y in range(stabilized_image.shape[1]):
            current_coordinates = np.array([[x], [y]])
            transformed_coordinates = np.add(np.dot(a, current_coordinates), b)
            new_x = transformed_coordinates.item(0)
            new_y = transformed_coordinates.item(1)
            stabilized_image[new_x, new_y] = image[x, y]

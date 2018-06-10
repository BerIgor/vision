import numpy as np
from scipy import interpolate
from hw3 import utils


def perform(frames, transformations):
    print("in q5.perform: number of frames: " + str(len(frames)))
    stabilized_images = list()
    for i in range(len(frames)):
        print("handling frame: " + str(i))
        frame = frames[i]
        a, b = transformations[i]
        stabilized_image = stabilize_image(frame, a, b)
        stabilized_images.append(stabilized_image)
        print(np.shape(stabilized_image))
    print("in q5.perform: number of stabilized frames " + str(len(stabilized_images)))
    return stabilized_images


def stabilize_image(image, a, b):
    """

    :param image: is the image to stabilize
    :param a: is the a matrix of an affine transformation
    :param b: is the b matrix of an affine transformation
    :return: a stabilized image
    """
    stabilized_image = np.zeros(np.shape(image))

    y_cord = np.array(list(range(stabilized_image.shape[0])))
    x_cord = np.array(list(range(stabilized_image.shape[1])))
    stabilized_coordinates = np.array([[x_cord], [y_cord]])
    print(stabilized_coordinates)
    print(np.shape(stabilized_coordinates))
    # Get the transformed coordinates: the coordinates in the ref image


    source_coordinates = utils.transform(stabilized_coordinates, a, b)
    print(np.shape(source_coordinates))
    for layer_index in range(image.shape[-1]):
        interpolation = interpolate.RectBivariateSpline(x, y, image[:, :, layer_index])

#        stabilized_image[:, :, layer_index] = interpolate(x_transformed, y_transformed)

    exit()
    x_coordinates = list()
    y_coordinates = list()
    for x in range(stabilized_image.shape[0]):
        for y in range(stabilized_image.shape[1]):
            current_coordinates = np.array([[x], [y]])
            transformed_coordinates = np.add(np.dot(a, current_coordinates), b)
            source_x = transformed_coordinates.item(0)
            source_y = transformed_coordinates.item(1)
            x_coordinates.append(source_x)
            y_coordinates.append(source_y)

    x = range(stabilized_image.shape[0])
    y = range(stabilized_image.shape[1])

    for layer_index in range(image.shape[-1]):
        # utils.cvshow("LAYER", image[:, :, layer_index])
        # stabilized_image = interpolate.interp2d(x_coordinates, y_coordinates, image[:, :, layer_index])
        interpolation = interpolate.RectBivariateSpline(x, y, image[:, :, layer_index])
        stabilized_image[:, :, layer_index] = interpolate()

    return stabilized_image

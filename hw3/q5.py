import numpy as np
from scipy import interpolate
from hw3 import utils
from scipy.interpolate import RegularGridInterpolator

def perform(frames, transformations):
    stabilized_images = list()
    for i in range(len(frames)):
        frame = frames[i]
        a, b = transformations[i]
        stabilized_image = stabilize_image(frame, a, b)
        stabilized_images.append(stabilized_image)
    return stabilized_images


def stabilize_image(image, a, b):
    """

    :param image: is the image to stabilize
    :param a: is the a matrix of an affine transformation
    :param b: is the b matrix of an affine transformation
    :return: a stabilized image
    """
    stabilized_image = interpolate_image_under_transformation(image, a, b)
    return stabilized_image


def interpolate_image_under_transformation(ref_image, a, b):
    req_image = np.zeros(np.shape(ref_image))

    rows = np.array(list(range(ref_image.shape[0])))
    cols = np.array(list(range(ref_image.shape[1])))
    stabilized_coordinates = np.array([[rows], [cols]])
    reference_coordinates = utils.transform(stabilized_coordinates, a, b)
    interpolator = RegularGridInterpolator((x, y, z), data)

    stabilized_r = list()
    stabilized_c = list()
    stabilized_coordinates = list()
    for l in range(ref_image.shape[2]):
        for r in range(ref_image.shape[0]):
            for c in range(ref_image.shape[1]):
                # This is row stacked, so we need to revert this when we get the values back
                current_coordinates = np.array([[r], [c]])
                stabilized_coordinates.append(current_coordinates)

        reference_coordinates = np.add(np.dot(a, stabilized_coordinates), b)
        RegularGridInterpolator(np.vstack(reference_coordinates). )



def interpolate_image_under_transformation2(ref_image, a, b):
    req_image = np.zeros(np.shape(ref_image))

    for layer in range((np.shape(req_image))[2]):
        for r, c in [(r, c) for r in range(req_image.shape[0]) for c in range(req_image.shape[1])]:
            req_point = utils.transform((r, c), a, b)
            req_c, req_r = req_point

            req_r = np.clip(req_r, 0, (np.shape(req_image))[0]-1)
            req_c = np.clip(req_c, 0, (np.shape(req_image))[1]-1)
            req_image[r, c, layer] = interpolate_point(ref_image[:, :, layer], (req_r, req_c))
    return req_image


def interpolate_point(ref_image, req_point):
    req_r, req_c = req_point
    req_r_low = np.int(np.floor(req_r))
    req_c_low = np.int(np.floor(req_c))
    req_r_high = np.int(np.ceil(req_r))
    req_c_high = np.int(np.ceil(req_c))

    # Legalize points
    req_r_low = np.clip(req_r_low, 0, (np.shape(ref_image))[0]-1)
    req_c_low = np.clip(req_c_low, 0, (np.shape(ref_image))[1]-1)
    req_r_high = np.clip(req_r_high, 0, (np.shape(ref_image))[0]-1)
    req_c_high = np.clip(req_c_high, 0, (np.shape(ref_image))[1]-1)

    r1 = (req_c_high - req_c)*ref_image[req_r_low, req_c_low] + (req_c - req_c_low)*ref_image[req_r_low, req_c_high]
    r2 = (req_c_high - req_c)*ref_image[req_r_high, req_c_low] + (req_c - req_c_low)*ref_image[req_r_high, req_c_high]
    value = (req_r_high - req_r)*r1 + (req_r - req_r_low)*r2

    return value

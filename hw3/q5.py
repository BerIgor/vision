import numpy as np
from scipy import interpolate
import cv2
from hw3 import utils
from scipy.interpolate import RegularGridInterpolator


def perform(frames, transformations):
    stabilized_images = list()
    for i in range(len(frames)):
        frame = frames[i]
        a, b = transformations[i]
        stabilized_image = stabilize_image_cv(frame, a, b)
        stabilized_images.append(stabilized_image)
    return stabilized_images


def stabilize_image_cv(frame, a, b):

    transformation_mat = np.hstack((a, b))
    transformed_frame = np.zeros((frame.shape[1], frame.shape[0], frame.shape[2]))

    for layer in range(frame.shape[2]):
        frame_layer = frame[:, :, layer]
        print("here")
        print(frame_layer.shape)

        curr_transformed_frame = cv2.warpAffine(src=frame_layer, M=transformation_mat,
                                                dsize=frame_layer.shape,
                                                flags=cv2.INTER_LINEAR)

        transformed_frame[:, :, layer] = curr_transformed_frame
        print("there")
        print(curr_transformed_frame.shape)

    return transformed_frame


def stabilize_image(image, a, b):
    """

    :param image: is the image to stabilize
    :param a: is the a matrix of an affine transformation
    :param b: is the b matrix of an affine transformation
    :return: a stabilized image
    """
    print(str(a) + " " + str(b))
    stabilized_image = interpolate_image_under_transformation(image, a, b)  # igor's method is 3
    return stabilized_image


"""
# TODO: This works, but you need to understand how
def interpolate_image_under_transformation(image, a, b):
    grid_x, grid_y = np.mgrid[range(image.shape[0]), range(image.shape[1])]
    z = np.array([grid_x.flatten(), grid_y.flatten()])
    zz = np.matmul(a, z) + b
    grid_x_new = np.reshape(zz[0, :], grid_x.shape)
    grid_y_new = np.reshape(zz[1, :], grid_y.shape)
    points = np.random.rand(image.shape[0] * image.shape[1], 2)
    points[:, 0] = grid_x_new.flatten()
    points[:, 1] = grid_y_new.flatten()
    img_stable = np.zeros(image.shape)
    for j in range(3):
        img_stable[:, :, j] = interpolate.griddata(points, image[:, :, j].flatten(), (grid_x, grid_y), method='linear')
    img_stable = np.uint8(img_stable)
    return img_stable
"""


def interpolate_image_under_transformation(image, a, b):
    grid_r, grid_c = np.mgrid[range(image.shape[0]), range(image.shape[1])]
    z = np.array([grid_r.flatten(), grid_c.flatten()])
    ai = np.linalg.inv(a)
    zz = np.matmul(ai, z - b)
    grid_r_new = np.reshape(zz[0, :], grid_r.shape)
    grid_c_new = np.reshape(zz[1, :], grid_c.shape)
    points = np.random.rand(image.shape[0] * image.shape[1], 2)
    points[:, 0] = grid_r_new.flatten()
    points[:, 1] = grid_c_new.flatten()
    img_stable = np.zeros(image.shape)
    for j in range(3):
        img_stable[:, :, j] = interpolate.griddata(points, image[:, :, j].flatten(), (grid_r, grid_c), method='linear')
    img_stable = np.uint8(img_stable)
    return img_stable


def interpolate_image_under_transformation0(ref_image, a, b):
    req_image = np.zeros(np.shape(ref_image))

    rows = np.arange(ref_image.shape[0])
    cols = np.arange(ref_image.shape[1])
    rv, cv = np.meshgrid(rows, cols, indexing='ij')

    reference_coordinates = np.stack((rv, cv), axis=2)
    # print(np.stack((rv, cv), axis=2))
    print(reference_coordinates)
    print(np.shape(reference_coordinates))
    exit()
    arr = np.reshape(reference_coordinates)
    print(arr)

    exit()
    grid = list()
    print(np.reshape(rv, (-1, 1)))
    print(np.reshape(cv, (-1, 1)))
    exit()
    # for i in range(size(rv))
    for r in np.reshape(rv, (-1, 1)):
        print("r==" + str(r))
        for c in np.reshape(cv, (-1, 1)):
            continue
    exit()

    for l in range(ref_image.shape[2]):
        current_layer = ref_image[:, :, l]
        interpolator = RegularGridInterpolator((rows, cols), current_layer,
                                               bounds_error=False, fill_value=0)
        for r in range(current_layer.shape[0]):
            for c in range(current_layer.shape[1]):
                # This is row stacked, so we need to revert this when we get the values back
                stabilised_coordinates = np.array([[r], [c]])
                reference_coordinates = utils.transform(stabilised_coordinates, a, b)
                req_image[r, c, l] = interpolator(reference_coordinates.transpose())

    return req_image


def interpolate_image_under_transformation2(ref_image, a, b):
    req_image = np.zeros(np.shape(ref_image))

    stabilized_r = list()
    stabilized_c = list()
    stabilized_coordinates = list()
    for l in range(ref_image.shape[2]):
        current_layer = ref_image[:, :, l]
        for r in range(current_layer.shape[0]):
            for c in range(current_layer.shape[1]):
                # This is row stacked, so we need to revert this when we get the values back
                current_coordinates = np.array([[r], [c]])
                stabilized_coordinates.append(current_coordinates)

        stabilized_coordinates = np.transpose(np.squeeze(stabilized_coordinates, 2))
        reference_coordinates = utils.transform(stabilized_coordinates, a, b)
        print(reference_coordinates)
        interpolator = RegularGridInterpolator(reference_coordinates, reference_coordinates, current_layer,
                                               bounds_error=False, fill_value=0)
        print(np.reshape(stabilized_coordinates, (1, -1)))
        interpolated_ = interpolator(np.reshape(stabilized_coordinates, (1, -1)))
        req_image[:, :, l] = interpolated_


# this works
def interpolate_image_under_transformation3(ref_image, a, b):
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

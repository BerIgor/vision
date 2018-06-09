import numpy as np
import cv2 as cv

# TODO: Complete q6: show 10 automatically matched points


def match_images(ref_image, ref_points, target_image, target_points, search_window, match_window):
    """
    Find the best matching point from target_points in ref_points
    :param ref_image: the reference image
    :param ref_points: feature points in the reference image
    :param target_image: the target image
    :param target_points: feature points in the target image
    :param search_window: half edge size of the search window - points outside will not be considered (see filter)
    :param match_window: half edge size of matching window for target_point matching
    :return:
    """
    matched_points = list()
    for ref_point in ref_points:
        points_in_window = filter_points_not_in_window(ref_point, target_points, search_window)
        template = get_sub_image(ref_image, ref_point, match_window)
        best_point = get_best_matching_point(template, target_image, points_in_window)
        matched_points.append(best_point)
    return matched_points


def get_best_matching_point(template, target_image, points):
    """
    Gets the point which is the best SSD match to template
    :param template: is the template to use
    :param target_image: is the image in which to calculate the SSD
    :param points: is a list of points (x,y)
    :return: returns a point with minimal SSD score
    """
    ssd_list = calc_ssd_list(template, target_image, points)
    min_index = ssd_list.index(min(ssd_list))
    return ssd_list[min_index]


def calc_ssd_list(template, image, points, match_window):
    """
    Calculate the Sum of Squared Differences in image using template, then for each point point in points save the
    minimal value in a sub-image defined by point and match_window
    :param template: the template used to calculate the ssd
    :param image: the image
    :param points: a list of points (x,y)
    :param match_window: (W in the assignment) is the window in which to search
    :return: returns a list with the minimal ssd values in each sub-image
    """
    ssd = cv.matchTemplate(image, template, cv.TM_SQDIFF)
    ssd_list = list()
    for point in points:
        window = get_sub_image(ssd, point, match_window)
        ssd_list.append(np.min(window))
    return ssd_list


def get_sub_image(image, point, window_size):
    """
    Gets a sub-image from image with size 2*window_size around point
    :param image: the image from which to extract sub-image
    :param point: the center point (x,y) of extraction window
    :param window_size: half the edge size of the window
    :return: a cut out from the image centered at point
    """
    x, y = point
    x_min = max(x - window_size, 0)
    y_min = max(y - window_size, 0)
    x_max = min(x + window_size, np.shape(image)[0])
    y_max = min(y + window_size, np.shape(image)[1])

    sub_image = image[x_min:x_max, y_min:y_max]
    return sub_image


def point_is_in_window(ref_point, point, search_window_size):
    """

    :param ref_point: is a point tuple (x,y) in the reference image
    :param point: is a list of point tuples (x,y) in some image
    :param search_window_size: is half the length of an edge for the window
    :return:
    """
    (ref_x, ref_y) = ref_point
    (p_x, p_y) = point
    x_dist = np.abs(ref_x - p_x)
    y_dist = np.abs(ref_y - p_y)
    if x_dist <= search_window_size and y_dist <= search_window_size:
        return True
    return False


def filter_points_not_in_window(ref_point, points, window_size):
    """
    Filters points so that they all are within the window
    A point is a tuple (x,y)
    :param ref_point: is the reference point
    :param points: is a list of points
    :param window_size: is half the size of the edge of the desired window
    :return: a filtered list
    """
    filtered_list = [point for point in points if point_is_in_window(ref_point, point, window_size) == True]
    return filtered_list

import numpy as np
import cv2 as cv
from hw3 import q2, utils


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
    filtered_ref_points = list()
    points_in_winow_num = list()
    for ref_point in ref_points:
        points_in_window = filter_points_not_in_window(ref_point, target_points, search_window)
        if not points_in_window:
            # Cancel reference feature points that no match was found for them
            continue
        filtered_ref_points.append(ref_point)
        points_in_winow_num.append(float(len(points_in_window)))
        if len(points_in_window) == 1:
            best_point = points_in_window[0]
        else:
            template = get_sub_image(ref_image, ref_point, match_window)
            # utils.cvshow("template", template)
            best_point = get_best_matching_point(template, target_image, points_in_window, match_window)
        matched_points.append(best_point)
    # print("ref points num: " + str(len(ref_points)) + "\nfiltered ref points num: " + str(len(filtered_ref_points))+ "\nAvg of feature points in matching window: " + str(sum(points_in_winow_num)/len(filtered_ref_points)))
    return filtered_ref_points, matched_points

def get_best_matching_point(template, target_image, points, match_window):
    """
    Gets the point which is the best SSD match to template
    :param template: is the template to use
    :param target_image: is the image in which to calculate the SSD
    :param points: is a list of points (x,y)
    :return: returns a point with minimal SSD score
    """
    ssd_list = calc_ssd_list(template, target_image, points, match_window)
    min_index = ssd_list.index(min(ssd_list))
    return points[min_index]


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
    # ssd = cv.matchTemplate(image, template, cv.TM_SQDIFF)
    ssd_list = list()
    for point in points:
        # window = get_sub_image(ssd, point, match_window)
        # ssd_list.append(np.min(window))
        image_ssd_sub_win = get_sub_image(image, point, match_window)
        # Truncate image according to template shape Image.shape must be >= template.shape
        template_new_rowmax = image_ssd_sub_win.shape[0] if template.shape[0] > image_ssd_sub_win.shape[0] else template.shape[0]
        template_new_colmax = image_ssd_sub_win.shape[1] if template.shape[1] > image_ssd_sub_win.shape[1] else template.shape[1]
        template = template[0:template_new_rowmax, 0:template_new_colmax]
        ssd_score = cv.matchTemplate(image_ssd_sub_win, template, cv.TM_SQDIFF)
        ssd_list.append(np.max(ssd_score)) # As it's calculated per window, min and max are the same

    return ssd_list


def get_sub_image(image, point, window_size):
    """
    Gets a sub-image from image with size 2*window_size around point
    :param image: the image from which to extract sub-image
    :param point: the center point (x,y) of extraction window
    :param window_size: half the edge size of the window
    :return: a cut out from the image centered at point
    """
    # utils.cvshow("original im", image)
    col, row = point  # x is col, y is row
    row_min = max(row - window_size, 0)
    col_min = max(col - window_size, 0)
    row_max = min(row + window_size, np.shape(image)[0])
    col_max = min(col + window_size, np.shape(image)[1])

    sub_image = image[row_min:row_max,col_min:col_max]
    # utils.cvshow("sub im", sub_image)

    return sub_image


def point_is_in_window(ref_point, point, search_window_size):
    """

    :param ref_point: is a point tuple (x,y) in the reference image
    :param point: is a list of point tuples (x,y) in some image
    :param search_window_size: is half the length of an edge for the window
    :return:
    """
    ref_x, ref_y = ref_point
    p_x, p_y = point
    x_dist = np.abs(ref_x - p_x)
    y_dist = np.abs(ref_y - p_y)
    return (x_dist <= search_window_size and y_dist <= search_window_size)


def filter_points_not_in_window(ref_point, points, window_size):
    """
    Filters points so that they all are within the window
    A point is a tuple (x,y)
    :param ref_point: is the reference point
    :param points: is a list of points
    :param window_size: is half the size of the edge of the desired window
    :return: a filtered list
    """
    filtered_list = list()
    for point in points:
        if point_is_in_window(ref_point, point, window_size):
            filtered_list.append(point)
    return filtered_list

def perform_q6(ref_image,target_image):
    """
    Receives a reference image and a target image and returns 10 automatically matched points between them (composition of all sub functions)
    :param ref_image: is the reference image
    :param target_image: is the target image
    :return: 2 lists of points. Pairs are matched by list index
    """

    # Window for optimal matching
    nms_window = 40 # Full Window size
    search_win = 20  # 0.5*L - Half Window size
    ssd_win = 5  # 0.5*W - Half Window size

    # Extract feature points for both images:
    ref_feature_points, ref_features_img = q2.harris_and_nms(ref_image, nms_window)
    target_feature_points, target_features_img = q2.harris_and_nms(target_image, nms_window)

    # utils.compare_two_images(ref_features_img, target_features_img, "Harris and nms - ref vs frame")

    return match_images(ref_image, ref_feature_points, target_image, target_feature_points, search_win, ssd_win)


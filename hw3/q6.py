import numpy as np
import cv2 as cv
from hw3 import q2, q3, utils


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
            best_point = get_best_matching_point(template, target_image, points_in_window, match_window)
        matched_points.append(best_point)

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
    ssd_list = list()

    for point in points:

        image_ssd_sub_win = get_sub_image(image, point, match_window)
        # Truncate image according to template shape Image.shape must be >= template.shape

        template_new_rowmax = image_ssd_sub_win.shape[0] if template.shape[0] > image_ssd_sub_win.shape[0] else template.shape[0]
        template_new_colmax = image_ssd_sub_win.shape[1] if template.shape[1] > image_ssd_sub_win.shape[1] else template.shape[1]

        template = template[0:template_new_rowmax, 0:template_new_colmax]

        ssd_score = cv.matchTemplate(image_ssd_sub_win, template, cv.TM_SQDIFF)
        ssd_list.append(np.min(ssd_score))  # As it's calculated per window, min and max are the same

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


def perform_q6(ref_image, target_image, mask, ref_points=None, search_win=5, ssd_win=40, nms_window=20):
    """
    Receives a reference image and a target image and returns 10 automatically matched points between them (composition of all sub functions)
    :param ref_image: is the reference image
    :param target_image: is the target image
    :param mask:
    :param ref_points:
    :return: 2 lists of points. Pairs are matched by list index
    """

    # Extract feature points for both images:
    if ref_points is None:
        ref_feature_points, ref_features_img = q2.harris_and_nms(ref_image, nms_window)
    else:
        print("predefined")
        ref_feature_points = ref_points

    target_feature_points, target_features_img = q2.harris_and_nms(target_image, nms_window)

    # filter out points that do not correspond to points in the mask
    f_ref_points, f_seq_points = filter_points_not_in_mask(ref_feature_points, mask, seq_points=target_feature_points)
    # marked_ref = q3.mark_points(ref_image, f_ref_points)
    # marked_seq = q3.mark_points(target_image, f_seq_points)
    # utils.compare_two_images(marked_ref, marked_seq, "comp")
    if len(f_ref_points) == 0:
        print("all points filtered")
        f_ref_points = ref_feature_points
        f_seq_points = target_feature_points
    # utils.compare_two_images(ref_features_img, target_features_img, "Harris and nms - ref vs frame")

    return match_images(ref_image, f_ref_points, target_image, f_seq_points, search_win, ssd_win)


def filter_points_not_in_mask(ref_points, mask, seq_points=None):
    new_ref_points = list()
    new_seq_points = list()
    for i in range(len(ref_points)):
        c, r = ref_points[i]
        if mask[r, c].any() > 0:
            new_ref_points.append(ref_points[i])
            if seq_points is not None:
                new_seq_points.append(seq_points[i])

    return new_ref_points, new_seq_points


def answer_question(frame_list):
    mask = cv.imread(utils.get_pwd() + '/our_data/masked_frames/0.jpg')
    mask = np.transpose(mask, (1, 0, 2))

    # ref_feature_points, ref_features_img = q2.harris_and_nms(frame_list[0], nms_window=10)
    # ref_feature_points, _ = filter_points_not_in_mask(ref_feature_points, mask)

    orig_ref_points = list()
    has_orig = False

    ref_frame_points_lists = list()
    seq_frame_points_lists = list()
    for i in range(1, len(frame_list)):
        ref_points, matched_points = perform_q6(frame_list[0], frame_list[i], mask,
                                                ssd_win=20,
                                                search_win=35,
                                                nms_window=35)

        if has_orig is False:
            orig_ref_points = ref_points
            orig_ref_points, _ = filter_points_not_in_mask(orig_ref_points, mask)
            has_orig = True

        ref_frame_points_lists.append(ref_points)
        seq_frame_points_lists.append(matched_points)
        orig_ref_points = _filter_points_not_in_points(orig_ref_points, ref_points)

    final_seq_points_lists = list()
    final_seq_points_lists.append(orig_ref_points)
    for i in range(len(ref_frame_points_lists)):
        seq_points_list = list()
        for ref_point in orig_ref_points:
            index = ref_frame_points_lists[i].index(ref_point)
            seq_points_list.append((seq_frame_points_lists[i])[index])

        final_seq_points_lists.append(seq_points_list)

    marked_frame_list = list()
    i = 0
    for point_list in final_seq_points_lists:
        f = frame_list[i].copy()

        marked_frame = q3.mark_points(f, point_list[0:10])
        marked_frame_list.append(marked_frame)
        i += 1

    marked_frame_list = utils.rotate_frames(marked_frame_list)

    # Flip horizontal
    new_frame_list = list()
    for image in marked_frame_list:
        image = np.flip(image, 1)
        new_frame_list.append(image)

    final = utils.hstack_frames(new_frame_list, reverse=False)

    cv.imwrite(utils.get_pwd() + "/q6" + ".jpg", final)


def _filter_points_not_in_points(ref_points_base, ref_points_current):
    """

    :param ref_points_base:
    :param ref_points_current:
    :return:
    """
    new_base_points = list()
    for base_point in ref_points_base:
        if base_point in ref_points_current:
            new_base_points.append(base_point)

    return new_base_points


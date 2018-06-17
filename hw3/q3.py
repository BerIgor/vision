import cv2 as cv
import numpy as np
from hw3 import utils, q2
from numpy import ascontiguousarray


# BGR colors
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
black = (0, 0, 0)
white = (255,255,255)


colors = [red, green, blue, black, white]

max_points = 6


def perform(frames):
    iml = list()
    frames_points = utils.get_frames_points()

    for i in range(len(frames)):
        frame = frames[i]
        points = frames_points[i]
        marked_frame = mark_points(frame, points)
        iml.append(marked_frame)

    iml = utils.rotate_frames(iml)
    iml = utils.hstack_frames(iml)
    cv.imwrite(utils.get_pwd() + "/q3" + ".jpg", iml)


def mark_points(image, points):
    """
    Draws shapes on provided points. Each points receives a unique shape
    @:param image is the image to mark
    @:param points is a list of coordinates where to draw shapes
            each coordinate is a tuple (x, y)
    @:returns the image with the points marked using shapes
    """
    # if len(points) > max_points:
    #     raise ValueError('More than 6 points in points')
    new_image = np.copy(image, 'C')
    for i in range(len(points)):
        option = int(i/len(colors))
        c_ind = i % len(colors)
        # print(points[i])
        if option == 0:
            new_image = np.ascontiguousarray(image, dtype=np.uint8)
            cv.circle(new_image, tuple(points[i]), 5, colors[c_ind], -1)
        else:
            top_left, bottom_right = get_rect(points[i])
            cv.rectangle(new_image, top_left, bottom_right, colors[c_ind], -1)

    return new_image


def get_rect(point, edge=5):
    """
    Returns the rect defining the rectangle centered at point with edge size edge
    :param point: the center of the rectangle (square)
    :param edge: size of the desired edge of rectangle
    :return: point tuples for top left and bottom right corners of the rectangle
    """
    top_left = tuple(np.subtract(point, edge))
    bottom_right = tuple(np.add(point, edge))
    return top_left, bottom_right


# def opencv_click_handle(event, x, y, flags, params):
#     # grab references to the global variables
#     global refPt
#     count = 0
#     # if the left mouse button was clicked, record the (x, y) coordinates
#
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         refPt.append((x, y))
#         cv.rectangle(ref_img, refPt[0], refPt[1], (0, 255, 0), 2)
#         print(count)
#         count += 1
#
#         # check if num of points was reached
#         # if count == max_points - 1:
#         #     marking = False


def choose_image_points(img,title):

    global refPt
    refPt = []
    title = "Mark points - " + title
    # Define click event
    def opencv_click_handle(event, x, y, flags, params):
        # grab references to the global variables
        count = 0
        # if the left mouse button was clicked, record the (x, y) coordinates

        if event == cv.EVENT_LBUTTONDBLCLK:
            click_coord = (x,y)
            refPt.append(click_coord)
            cv.circle(img, click_coord, 5, red)
            print(count)
            count = count + 1

            # check if num of points was reached
            # if count == max_points - 1:
            #     marking = False

    cv.namedWindow(title)
    cv.setMouseCallback(title, opencv_click_handle)
    num_points = 6

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv.imshow(title, img)
        key = cv.waitKey(1) & 0xFF

        # if the 'c' key is pressed, break from the loop
        if key == ord("q"):
            break

    # mark_points(img,refPt)

    return refPt

def choose_match_points_for_all_frames(frame_list):
    # Reference point marking
    ref_image = frame_list[0]
    image_harris_nms = q2.harris_and_nms(ref_image)
    marked_ref_plist = choose_image_points(image_harris_nms, "Reference Image")
    utils.cvshow("Reference Image marked",mark_points(ref_image, marked_ref_plist))

    # Choose points for all images
    marked_points_in_all_frames = []
    for i in range(len(frame_list[1:])):
        frame = frame_list[i+1]
        title = "Frame " + str(i+2)
        # Harris and NMS for each frame:
        frame_harris_nms = q2.harris_and_nms(frame)
        utils.cvshow(title + " harris results",frame_harris_nms)
        # # Choose match points for reference
        # print("Choosing points for " + title)
        # marked_ref_plist = q3.choose_image_points(frame_harris_nms, title)
        # marked_points_in_all_frames.append(marked_ref_plist)
        # q3.mark_points(image_harris_nms, marked_ref_plist)
        # # close all open windows
        # cv.destroyAllWindows()


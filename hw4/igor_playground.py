import numpy as np
import cv2
import sklearn.preprocessing
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from hw4 import utils
from hw4 import style_transfer
from hw4 import texture_transfer
from hw4 import ariel_playground


def get_indices_to_keep(frame_list):
    # calc laplacian of all frames
    var_list = list()
    for frame in frame_list:
        var = cv2.Laplacian(frame, cv2.CV_64F).var()
        var_list.append(var)

    ind = np.argpartition(var_list, 5)[5:]
    ind = sorted(ind)

    # filtered_frame_list = [frame_list[i] for i in ind]

    return ind


def pixel_move(image, image_lap, point):
    x, y = point

    d_section = utils.get_sub_image(image_lap, point, 1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(d_section)  # the points returned are in r,c format

    r, c = max_loc
    r -= 1
    c -= 1

    r = y + r
    c = x + c

    return image[r, c, :]


def image_move(image):
    # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # lap = cv2.Laplacian(grey, cv2.CV_64F)

    moved_image = image.copy()

    grey = cv2.cvtColor(moved_image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    for x in range(1, image.shape[1] - 1):
        for y in range(1, image.shape[0] - 1):
            moved_image[y, x, :] = pixel_move(image, lap, (x, y))
        # cv2.imwrite(utils.get_pwd() + '/our_results/image_move' + str(i) + '.jpg', moved_image)

    return moved_image


def calc_motion_mat(image):

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_64F)

    movement_mat = np.zeros((image.shape[0], image.shape[1], 2))
    for x in range(1, image.shape[1] - 1):
        for y in range(1, image.shape[0] - 1):
            d_section = utils.get_sub_image(lap, (x, y), 1)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(d_section)  # the points returned are in r,c format

            r, c = max_loc
            r -= 1
            c -= 1

            r = y + r
            c = x + c
            movement_mat[y, x, 0] = r
            movement_mat[y, x, 1] = c
    return movement_mat


def image_move2(image, motion_matrix):
    moved_image = image.copy()
    for x in range(1, image.shape[1] - 1):
        for y in range(1, image.shape[0] - 1):
            src_r = int(motion_matrix[y, x, 0])
            src_c = int(motion_matrix[y, x, 1])
            moved_image[y, x, :] = image[src_r, src_c, :]
    return moved_image


def calc_bg_motion_frames(image, frames=5):
    motion_frames_list = list()
    moved_bg = image.copy()
    for i in range(frames):
        # motion_mat = calc_motion_mat(moved_bg)
        # moved_bg = image_move2(moved_bg, motion_mat)
        moved_bg = image_move(moved_bg)
        motion_frames_list.append(moved_bg)

    return motion_frames_list


if __name__ == "__main__":

    full_frame_list = utils.get_all_frames(utils.get_pwd() + '/our_data/ariel.avi')
    full_frame_mask_list = utils.get_all_frames(utils.get_pwd() + '/our_data/mask.avi')
    texture = cv2.imread(utils.get_pwd() + '/our_data/style.jpg')
    rows, cols, _ = np.shape(full_frame_list[0])

    background = cv2.imread(utils.get_pwd() + '/our_data/starry_bg.jpg')
    # background = background[:rows, :cols, :]
    bg_motion_list = calc_bg_motion_frames(background)
    new_bg_motion_list = list()
    for bg_frame in bg_motion_list:
        for i in range(15):
            new_bg_motion_list.append(bg_frame)

    bg_motion_list = new_bg_motion_list
    reversed_bg_motion_list = bg_motion_list.copy()
    reversed_bg_motion_list.reverse()
    bg_motion_list.extend(reversed_bg_motion_list)

    ind = get_indices_to_keep(full_frame_list)  # Filter blurred frames
    full_frame_list = [full_frame_list[i] for i in ind]
    full_frame_mask_list = [full_frame_mask_list[i] for i in ind]

    frames_features = ariel_playground.detect_features(full_frame_list)

    frame_0_points = frames_features[0]
    final_frames_list = list()
    bg_index = 0
    M_list = list()
    for i in range(len(full_frame_list)):
        # Handle the frame
        if len((frames_features[i])) > 3:
            continue
        M = cv2.getAffineTransform(np.float32(frames_features[i]), np.float32(frame_0_points))
        stabilized_frame = cv2.warpAffine(full_frame_list[i], M, (cols, rows))

        M_list.append(M[0, 2])

        background_moved = bg_motion_list[bg_index % len(bg_motion_list)]
        M_bg = M.copy()
        M_bg[:, 0:2] = np.array([[1, 0], [0, 1]])
        M_bg[1, 2] = np.array([0])
        background_moved = cv2.warpAffine(background_moved, M_bg, (cols, rows))
        background_moved = background_moved[:rows, :cols, :]

        # _, mask = cv2.threshold(full_frame_mask_list[i], 127, 1, cv2.THRESH_BINARY)
        stabilized_mask = cv2.warpAffine(full_frame_mask_list[i], M, (cols, rows))
        _, stabilized_mask = cv2.threshold(stabilized_mask, 127, 1, cv2.THRESH_BINARY)

        fg = np.multiply(stabilized_mask, stabilized_frame)
        mask_inv = 1 - stabilized_mask
        bg = np.multiply(mask_inv, background_moved)

        new_frame = fg + bg

        final_frames_list.append(new_frame)
        bg_index += 1

    utils.make_normal_video(utils.get_pwd() + '/our_results/combined.avi', final_frames_list)

    plt.plot(range(len(final_frames_list)), M_list, 'ro')
    plt.show()
    ransac_regressor = linear_model.RANSACRegressor()
    frame_nums = [i for i in range(len(final_frames_list))]
    frame_nums = np.reshape(frame_nums, (1, -1))

    M_array = np.reshape(np.asarray(M_list), (1, -1))

    ransac_regressor.fit(frame_nums, M_array)
    predict = ransac_regressor.predict(frame_nums)

    plt.plot(frame_nums.tolist(), predict.tolist(), 'ro')
    plt.show()
    exit()


    new_frame_list = list()

    background_moved = background
    for i in range(min(len(full_frame_mask_list), len(full_frame_list))):

        # motion_mat = calc_motion_mat(background_moved)
        # gather images
        # background_moved = image_move2(background_moved, motion_mat)
        # background_moved = image_move(background_moved)
        background_moved = bg_motion_list[i % len(bg_motion_list)]
        # background_moved = cv2.GaussianBlur(background_moved, ksize=(5, 5), sigmaX=0, sigmaY=0)
        frame = full_frame_list[i]
        mask = full_frame_mask_list[i]

        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        var = cv2.Laplacian(frame, cv2.CV_64F).var()
        if var < 50:
            continue  # skip this frame

        # handle masking
        fg = np.multiply(mask, frame)
        mask_inv = 1 - mask
        bg = np.multiply(mask_inv, background_moved)

        new_frame = fg + bg
        new_frame_list.append(new_frame)

    utils.make_normal_video(utils.get_pwd() + '/our_results/combined2.avi', new_frame_list)

    exit()
    background_resized = cv2.resize(background, dsize=(cols, rows))
    background_resized = background_resized / 255
    # background_resized = cv2.GaussianBlur(background_resized, ksize=(31, 31), sigmaY=0, sigmaX=0)

    texture_resized = cv2.resize(texture, dsize=(cols, rows))
    # texture_resized = cv2.GaussianBlur(texture_resized, ksize=(11, 11), sigmaX=0, sigmaY=0)
    texture_resized = texture_resized / 255
    texture_resized = np.clip(texture_resized, a_max=1.0, a_min=0.0)

    frame_resized = cv2.resize(frame, dsize=(cols, rows))
    # texture_resized = cv2.GaussianBlur(texture_resized, ksize=(11, 11), sigmaX=0, sigmaY=0)
    frame_resized = frame_resized / 255
    frame_resized = np.clip(frame_resized, a_max=1.0, a_min=0.0)
    # bg_path = utils.get_pwd() + '/our_data/starry_bg.jpg'
    # te_path = utils.get_pwd() + '/our_data/style.jpg'
    # tt = texture_transfer.TextureTransferTool(te_path, bg_path, 64, 64, 3, 0.3, 0.3, 0.8, 0.005, 0)
    # res = tt.start(utils.get_pwd() + '/res.jpg')

    res = style_transfer.transfer(background_resized, texture_resized)
    # utils.cvshow("orig", background)
    utils.cvshow("res", res)

    exit()




    counter = 0
    for frame in full_frame_list:
        var = cv2.Laplacian(frame, cv2.CV_64F).var()
        if var < 50:
            counter += 1

    print(counter)
    """
    frame = full_frame_list[0]
    face_template = cv2.imread(utils.get_pwd() + '/our_data/face_template.bmp')
    res = cv2.matchTemplate(frame, face_template, cv2.TM_SQDIFF_NORMED)
    utils.cvshow("res", res)
    """
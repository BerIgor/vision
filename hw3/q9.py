import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from hw3 import utils, q7, q5, q3


def perform_subspace_video_stabilization(frame_list, output_video_path):
    # Stages are according to the moodle note - https://moodle.technion.ac.il/mod/forum/discuss.php?d=423166

    # Stage 1 - Extract KLT features and build matrix M:
    M = extract_klt_features(frame_list)
    print("Number of zeros in M: " + str(M.size-np.count_nonzero(M)))
    # Plot trajectory matrix M
    # Create custom binary colormap
    # cmap = plt.cm.Greys
    # cmaplist = [(0, 0, 1, 1) for i in range(cmap.N)]
    # cmaplist.insert(0, ( 1, 1, 1, 1))
    # cmap = cmap.from_list('Custom cmap', cmaplist, len(cmaplist))
    # # Plot M
    # plt.matshow(M != 0, cmap=cmap)
    # plt.show()

    # Stage 2 - Break M to windows
    frames_per_window = 50
    window_delta = 5
    M_windows_list = break_m_into_windows(M, frames_per_window, window_delta)

   # Stage 3 - Create truncated window list
    M_windows_list_truncated = truncate_zeros_from_window_list(M_windows_list)

    # Stage 4 - Smoothing and filtering using SVD and a guassian 1D temporal filter
    M_windows_list_truncated_smooth = smooth_and_filter(M_windows_list_truncated)

    # Stage 5 - Get transformation for each frame using RANSAC
    frames_list_transformations = get_frames_transformations(M_windows_list_truncated, M_windows_list_truncated_smooth, window_delta)

    # Stage 6 - Stabilize all frames using transformations found in stage 5
    frame_list_stabilized = stabilize_frames(frame_list, frames_list_transformations)

    # Stage 7 - Rebuild the entire stabillized video
    if os.path.isfile(output_video_path):
        os.remove(output_video_path)
    utils.make_normal_video(output_video_path, frame_list_stabilized)

    return frame_list_stabilized


def extract_klt_features(frame_list):
    # Based on: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
    # Some parts of the template code was preserved in comments to allow results visualization later on, if needed
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=frame_list[0].size,
                          qualityLevel=0.01,
                          minDistance=30,
                          blockSize=3)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors (legacy, from template code)
    # color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    old_frame = frame_list[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    print("Num of features found: " + str(p0.shape[0]))
    M = np.zeros((old_frame.size*2,len(frame_list))) # Initialize M matrix in a way that it can fit new detected features in future frames
    p0_xy_1d_vec = p0[:, 0, :].flatten()
    # Plot features on frame
    # p0_tuples_list = xy_vec_to_tuples_list(p0_xy_1d_vec)
    # utils.cvshow("goodFeaturesToTrack", q3.mark_points(old_frame.copy(), p0_tuples_list))
    M[0:p0_xy_1d_vec.size, 0] = p0_xy_1d_vec
    # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)
    for i in range(len(frame_list[1:])):
        frame_gray = cv2.cvtColor(frame_list[i+1], cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if st.size != np.count_nonzero(st):
            # Zero out features that couldn't be tracked
            p1[np.where(st == 0)[0], 0, :] = -1
            # print("Lost track of " + str(st.size - np.count_nonzero(st)) + " features")

        # Extract new features
        frame_new_features = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        print("num of new points: " + str(frame_new_features.shape[0]))

        # Filter out new found features that are already in p1
        win_size = 20
        frame_new_features_xy_2d_prev = frame_new_features[:, 0, :]
        for p1_point in p1[:, 0, :]:
            frame_new_features_filtered = [new_p for new_p in frame_new_features_xy_2d_prev if (abs(p1_point[0]-new_p[0]) > win_size or abs(p1_point[1]-new_p[1]) > win_size)]
            frame_new_features_xy_2d_prev = frame_new_features_filtered
        if len(frame_new_features_filtered) > 0:
            print("New features: " + str(len(frame_new_features_filtered)))
        frame_new_features_xy_1d_vec = np.array(frame_new_features_filtered).flatten()

        # Select good points (legacy, from template code)
        # good_new = p1[st == 1]
        # good_old = p0[st == 1]

        # Concatenate p1 and new features and assign to M
        p1_xy_1d_vec = p1[:, 0, :].flatten()
        new_points_1d_xy_vec = np.concatenate([p1_xy_1d_vec, frame_new_features_xy_1d_vec]) if frame_new_features_xy_1d_vec.size > 0 else p1_xy_1d_vec
        M[0:new_points_1d_xy_vec.size,i+1] = new_points_1d_xy_vec

        # draw the tracks (legacy, from template code)
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        # img = cv2.add(frame, mask)
        # cv2.imshow('frame', img)
        # k = cv2.waitKey(27) & 0xff
        # if k == 27:
        #     break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = new_points_1d_xy_vec.reshape((int(new_points_1d_xy_vec.size/2), 2)).reshape(-1, 1, 2) # First 1d -> 2d reshape, then 2d -> 3d standard calcOpticalFlowPyrLK I/O form reshape

    cv2.destroyAllWindows()
    # Remove redundant zero rows from m
    M_no_zeroes =  M[0,:].copy()
    for row in M[1:,:]:
        if np.count_nonzero(row) == 0:
            break
        M_no_zeroes = np.vstack((M_no_zeroes,row))

    M_no_zeroes[M_no_zeroes == -1] = 0 # remove st == 0 placeholders

    return M_no_zeroes


def break_m_into_windows(M, frames_per_window, window_delta):
    m_windows_list = list()
    num_of_frames = M.shape[1]
    windows_num = math.floor((num_of_frames-frames_per_window) / window_delta)
    print("Number of windows in M: " + str(windows_num) + " for window size of K=" + str(frames_per_window) + " frames")
    for i in range(windows_num):
        curr_win_start = window_delta * i
        curr_win_end = curr_win_start + frames_per_window

        curr_window = M[:, curr_win_start:curr_win_end]
        m_windows_list.append(curr_window)

    last_win_start = windows_num*window_delta
    last_win_end = num_of_frames
    last_window = M[:, last_win_start:last_win_end]
    m_windows_list.append(last_window)

    return m_windows_list


def truncate_zeros_from_window_list(window_list):
    window_list_truncated = list()
    for window in window_list:
        if np.count_nonzero(window) == window.size:
            # No zeros in window:
            window_list_truncated.append(window)
        else:
            truncated_win = window[~np.all(window == 0, axis=1)]
            window_list_truncated.append(truncated_win)

    return window_list_truncated


def smooth_and_filter(truncated_window_list):
    from numpy.linalg import svd
    from scipy.signal import savgol_filter as filter
    from scipy.ndimage.filters import gaussian_filter1d

    truncated_smoothed_window_list = list()

    for win in truncated_window_list:
        r = 9 # Irani [2002]

        u, s, v = svd(win)
        # Create coefficient matrix C and eigen-trajectories matrix E
        s_diag = np.diag(np.sqrt(s))
        s_diag_r = s_diag[:r, :r]
        c = np.dot(u[:, :r], s_diag_r)
        e = np.dot(s_diag_r, v[:r, :])
        # e_stab = filter(e,window_length=5,polyorder=2,axis=1) # Savitzky-Golay filter can also be applied, instead of the gaussian
        e_stabilized = gaussian_filter1d(e, sigma=(win.shape[1]/2)/(2**0.5), axis=1) # Sigma is as suggested in the paper
        smoothed_win = np.dot(c, e_stabilized)

        # Show norm difference before and after stabilization on frame 0
        # print("Diff on frame 1 after smoothing:")
        # print(np.linalg.norm(win[:, 0]-smoothed_win[:, 0]))

        truncated_smoothed_window_list.append(smoothed_win)

    return truncated_smoothed_window_list


def ransac_on_windows(win, smooth_win):

    # Get dimensions - identical for both windows
    if win.shape != smooth_win.shape:
        raise ValueError("Input windows must have the same shape")
    num_of_frames = win.shape[1]
    num_of_points = win.shape[0]
    transformations_list = list()

    for i in range(num_of_frames):
        win_point_list = win[:, i]
        smooth_win_point_list = smooth_win[:, i]
        # Fit arrays to ransac input:
        win_points_ransac_fitted = [(win_point_list[i], win_point_list[i+1]) for i in range(0,num_of_points,2)]
        smooth_win_points_ransac_fitted = [(smooth_win_point_list[i], smooth_win_point_list[i+1]) for i in range(0, num_of_points, 2)]
        # Get transformation using RANSAC from q7
        A, b = q7.calc_transform_ransac(smooth_win_points_ransac_fitted, win_points_ransac_fitted)
        transformations_list.append((A,b))

    return transformations_list


def get_frames_transformations(M_windows_list_truncated, M_windows_list_truncated_smooth, window_delta):
    # Initialize
    frames_transformations_list = list()
    num_of_windows = len(M_windows_list_truncated)
    # Get relevant frames from window
    for i in range(num_of_windows):
        if i != num_of_windows - 1:
            # Take only the first 5 frames
            truncated_win = M_windows_list_truncated[i][:, 0:window_delta]
            truncated_win_smooth = M_windows_list_truncated_smooth[i][:, 0:window_delta]
        else:
            # Last window - take all of it
            truncated_win = M_windows_list_truncated[i]
            truncated_win_smooth = M_windows_list_truncated_smooth[i]

        # Extract transformation matrices using RANSAC
        print("Calculating transformation for window " + str(i) + " with " + str(truncated_win.shape[1]) + " frames")
        window_transformation_list = ransac_on_windows(truncated_win, truncated_win_smooth)

        # Concatenate transformations list
        frames_transformations_list = window_transformation_list if len(frames_transformations_list) == 0 else frames_transformations_list + window_transformation_list

    return frames_transformations_list


def stabilize_frames(frame_list, frames_transformations_list, save_frames=False):
    # Initialize
    stabilized_frames = list()
    num_of_frames = len(frame_list)

    # Clean output directories
    pwd = utils.get_pwd()
    images_dir_path = pwd + '/our_data/q9_frames/'
    stab_images_dir_path = pwd + '/our_data/q9_frames_stab/'
    if save_frames:
        clean_output_directories(images_dir_path, stab_images_dir_path)

    # Transform frames
    for i in range(num_of_frames):
        a, b = frames_transformations_list[i]
        # print("Transformation matrices for frame " + str(i))
        # print(a)
        # print(b)
        frame = frame_list[i]
        stabilized_frame = q5.stabilize_image(frame, a, b)
        stabilized_frames.append(stabilized_frame)
        # Save frames to output folders
        if save_frames:
            utils.video_save_frame(frame, pwd, 'q9_frames', i)
            utils.video_save_frame(stabilized_frame, pwd, 'q9_frames_stab', i)
        print("Stabilized frame " + str(i))

    return stabilized_frames

# Utility functions for q9
def xy_vec_to_tuples_list(xy_vec):
    return [(xy_vec[i], xy_vec[i + 1]) for i in range(0, xy_vec.size, 2)]


def clean_output_directories(images_dir_path, stab_images_dir_path=None):
    import shutil

    # Delete and re-create current output folders
    if os.path.isdir(images_dir_path):
        shutil.rmtree(images_dir_path)
    try:
        original_umask = os.umask(0)
        os.makedirs(images_dir_path)
    finally:
        os.umask(original_umask)

    if stab_images_dir_path is not None:
        if os.path.isdir(stab_images_dir_path):
            shutil.rmtree(stab_images_dir_path)
        try:
            original_umask = os.umask(0)
            os.makedirs(stab_images_dir_path)
        finally:
            os.umask(original_umask)




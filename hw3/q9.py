import numpy as np
import cv2
import math
from hw3 import utils, q7, q5, program

def perform_subspace_video_stabilization(frame_list):
    # Stages are according to the moodle note - https://moodle.technion.ac.il/mod/forum/discuss.php?d=423166

    # Stage 1 - Extract KLT features and build matrix M:
    M = extract_klt_features(frame_list)

    # Stage 2 + 3 - Break M to windows and truncate zeros from window
    frames_per_window = 50
    window_delta = 5
    M_windows_list = break_m_into_windows(M, frames_per_window, window_delta)

   # Stage 3 - Create truncated window list
    M_windows_list_truncated = truncate_zeros_from_window_list(M_windows_list)

    # # Stage 4 - Smoothing using SVD and filtering
    M_windows_list_truncated_smooth = smooth_and_filter(M_windows_list_truncated)

    # Stage 5 + 6 - Get transformation for each frame using RANSAC
    frames_transformations_list = list()
    for i in range(M_windows_list_truncated_smooth):
        if i != len(M_windows_list_truncated_smooth)-1:
            # Take only the first 5 frames
            truncated_win = M_windows_list_truncated[i][:, 0:window_delta-1]
            truncated_win_smooth = M_windows_list_truncated_smooth[i][:, 0:window_delta-1]
        else:
            # Last window - take all of it
            truncated_win = M_windows_list_truncated[i]
            truncated_win_smooth = M_windows_list_truncated_smooth[i]

        # Extract transformation matrices using RANSAC
        window_transformation_list = ransac_on_windows(truncated_win, truncated_win_smooth)

        # Concatenate transformations list
        frames_transformations_list = window_transformation_list if len(frames_transformations_list) == 0 else frames_transformations_list + window_transformation_list


    # Stage 6 - Stabilize all frames using stage 5 transformations
    stabilized_frames = list()
    num_of_frames = len(frame_list)
    for i in range(num_of_frames):
        a, b = frames_transformations_list[i]
        stabilized_frames.append(q5.stabilize_image(frame_list[i], a, b))

    # Stage 7 - Rebuild the entire stabillized video
    # output_video_path = pwd + '/our_data/ariel_stabilized_q9.mp4'
    # program.make_normal_video(output_video_path, stabilized_frames)


def extract_klt_features(frame_list):
    # Based on: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
    # TODO - parameters may vary for our video
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=frame_list[0].size,
                          qualityLevel=0.05,
                          minDistance=3,
                          blockSize=5)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    old_frame = frame_list[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    print("Num of features found: " + str(p0.shape[0]))
    M = np.zeros((p0.shape[0]*2,len(frame_list))) # Initialize M matrix
    M[:, 0] = p0[:, 0, :].flatten()
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    for i in range(len(frame_list)-1):
        frame_gray = cv2.cvtColor(frame_list[i+1], cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # if st.size != np.count_nonzero(st):
        #     print("iter " + str(i))
        # Select good points
        good_new = p1[st == 1] # TODO - Do we need this? it can filter points, which can cause issues. Alternatively, we can pad with zeros
        good_old = p0[st == 1]
        M[:,i+1] = good_new.flatten()
        # draw the tracks
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
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    return M


def break_m_into_windows(M, frames_per_window, window_delta):
    m_windows_list = list()
    num_of_frames = M.shape[1]
    windows_num = math.floor((num_of_frames-frames_per_window) / window_delta)
    print(windows_num)
    for i in range(windows_num):
        curr_win_start = window_delta * i
        curr_win_end = curr_win_start + frames_per_window-1
        print("Current window range:")
        print([curr_win_start, curr_win_end])
        curr_window = M[:, curr_win_start:curr_win_end]
        # curr_window_truncated = truncate_zeros_from_window(curr_window)
        m_windows_list.append(curr_window)

    last_win_start = windows_num*window_delta
    last_win_end = num_of_frames-1
    last_window = M[:, last_win_start:last_win_end]
    # last_window_truncated = truncate_zeros_from_window(last_window)
    m_windows_list.append(last_window)

    return m_windows_list

def truncate_zeros_from_window_list(window_list):
    # TODO - Check why there are no zeros in M
    # print("Num of zeros in window: " + str(window.size-np.count_nonzero(window)))
    window_list_truncated = list()
    for window in window_list:
        if np.count_nonzero(window) == window.size:
            # No zeros in window:
            window_list_truncated.append(window)
        else:
            truncated_win = np.zeros(2, window.shape[1])
            for i in range(0, window.shape[0], 2):
                xy_rows = window[i:i + 1, :]
                if np.count_nonzero(xy_rows) == xy_rows.size:
                    # Leave only rows where no zeros exist, i.e. features do not disappear during window
                    if np.count_nonzero(truncated_win) == 0:
                        # No rows has been entered yet
                        truncated_win = xy_rows
                    else:
                        np.vstack((truncated_win, xy_rows))
            window_list_truncated.append(truncated_win)

    return window_list_truncated


def smooth_and_filter(truncated_window_list):
    from sklearn.utils.extmath import randomized_svd as rsvd
    from scipy.signal import savgol_filter as filter

    truncated_smoothed_window_list = list()
    for win in truncated_window_list:
        r = 9 # As suggested by Irani [2002]
        u, s, vh = rsvd(win, n_components=r)
        c = np.matmul(u, np.diag(s))
        e = vh

        # Filtering
        e_stab = np.zeros_like(e)
        e_stab[r-1, :] = filter(e[r-1, :], window_length=25, polyorder=2)
        truncated_smoothed_window_list.append(np.matmul(c, e_stab)) # result is win stabilized

def ransac_on_windows(win, smooth_win):

    # Get dimensions - identical for both windows
    if win.shape != smooth_win.shape:
        raise ValueError("Input windows must have the same shape")
    num_of_frames = win.shape[1]
    num_of_points = win.shape[0]

    transformations_list = list()

    for i in range(num_of_frames):
        win_point_list = win[:, i]
        smooth_win_point_list = smooth_win[:, i] # TODO - is this a 1D vec or 2D
        # Fit arrays to ransac input:
        win_points_ransac_fitted = [(win_point_list[i], win_point_list[i+1]) for i in range(0,num_of_points,2)]
        smooth_win_points_ransac_fitted = [(smooth_win_point_list[i], smooth_win_point_list[i+1]) for i in range(0,num_of_points,2)]
        # Apply ransac
        A, b = q7.calc_transform_ransac(win_points_ransac_fitted,smooth_win_points_ransac_fitted)
        transformations_list.append((A,b))

    return transformations_list


if __name__ == "__main__":
    # Test q9
    import os
    pwd = os.getcwd().replace('\\', '//')
    source_video_path = pwd + '/our_data/ariel.mp4'
    all_video_frames = utils.get_all_video_frames(source_video_path)
    perform_subspace_video_stabilization(all_video_frames)



import numpy as np
import cv2
import math

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
    # M_windows_list_truncated_smooth = smooth_and_filter(M_windows_list_truncated)
    #
    # # Stage 5 + 6 - RANSAC and Transform on each window - 5 frames for each window except last
    # Stabillized_result = list()
    # for i in range(M_windows_list_truncated_smooth):
    #     if i != len(M_windows_list_truncated_smooth)-1:
    #         # Take only the first 5 frames
    #         tuncated_win = M_windows_list_truncated[i][:, 0:window_delta-1]
    #         tuncated_win_smooth = M_windows_list_truncated_smooth[i][:, 0:window_delta-1]
    #     else:
    #         # Last window - take all of it
    #         tuncated_win = M_windows_list_truncated[i]
    #         tuncated_win_smooth = M_windows_list_truncated_smooth[i]
    #
    #     # Extract transformation matrices using RANSAC
    #     A,b = ransac_on_windows(tuncated_win, tuncated_win_smooth) # TODO - Implement function that finds A,b matrcies from win to smooth
    #
    #     # Created stabillized version of truncated window
    #     # TODO - do we need to do this on the original frames or on the windows list (and if so, which list? [reg/truncated])
    #     # Stabillized_result.append(stabilize_window(tuncated_win, A, b))
    #     # OR
    #     # Stabillized_result.append(stabilize_frame(frame_list[5*i:5*(i+1)])) # TODO - Not sure correct indices are taken
    #
    # # Stage 7 - Rebuild the entire stabillized video
    # # TODO - use q1_make_video from porgram.py or just do this in main



def extract_klt_features(frame_list):
    # Based on: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
    # TODO - parameters may vary for our video
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=frame_list[0].size,
                          qualityLevel=0.1,
                          minDistance=3,
                          blockSize=7)
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
        print(i)
        frame_gray = cv2.cvtColor(frame_list[i+1], cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if st.size != np.count_nonzero(st):
            print("iter " + str(i))
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
    windows_num = math.floor(num_of_frames / window_delta)
    for i in range(windows_num):
        print("Window " + str(i))
        next_ind = frames_per_window + window_delta * i
        curr_window = M[:, window_delta * i:next_ind]
        # curr_window_truncated = truncate_zeros_from_window(curr_window)
        m_windows_list.append(curr_window)

    last_window = M[:, next_ind:num_of_frames-1]
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
        r = truncated_window_list.shape[0]
        u, s, vh = rsvd(win, n_components=r)
        c = np.matmul(u, np.diag(s))
        e = vh

        # Filtering
        e_stab = np.zeros_like(e) # TODO - probably wrong, I think its smaller
        e_stab[r, :] = filter(e[r, :])
        truncated_smoothed_window_list.append(np.matmul(c, e_stab)) # result is win stabilized



## Igor functions start here (Stages 5-9) ###


import numpy as np
import cv2
from hw4 import utils
from hw3 import q3


def is_point_in_rect(x,y,w,h,p_x,p_y):
    return p_x > x and p_x < x + w and p_y > y and  p_y < y + h


def detect_features():
    pwd = utils.get_pwd()

    # Input
    # frame_list = utils.get_frames_uniform(utils.get_pwd() + '/our_data/ariel.avi', 13)
    frame_list = utils.get_all_frames(utils.get_pwd() + '/our_data/ariel.avi')
    # image = frame_list[130]
    haar_dir = pwd + "/haar_xmls"
    faceCascPath = haar_dir + "/haarcascade_frontalface_default.xml"
    eyeCascPath = haar_dir + "/haarcascade_eye.xml"
    noseCascPath = haar_dir + "/Nariz_nose.xml"
    mouthCascPath = haar_dir + "/Mouth.xml"

    # Output
    haar_dir_results = pwd + "/our_results/haar_detection"
    utils.clean_output_directories(haar_dir_results)

    # Construct the haar cascades
    faceCascade = cv2.CascadeClassifier(faceCascPath)
    eyeCascade = cv2.CascadeClassifier(eyeCascPath)
    noseCascade = cv2.CascadeClassifier(noseCascPath)
    mouthCascade = cv2.CascadeClassifier(mouthCascPath)

    # Counters
    no_nose_cnt = 0
    bad_frames_cnt = 0
    very_bad_frames_cnt = 0

    frames_features = list()
    for i in range(len(frame_list)):
        # Read the image
        image = frame_list[i]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect features in the image
        feature_params = dict(maxCorners=gray.size, qualityLevel=0.05, minDistance=5, blockSize=3)
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        p0_1d = p0[:, 0, :].flatten()
        p0_tuples = utils.xy_vec_to_tuples_list(p0_1d)
        # utils.cvshow("goodFeaturesToTrack", q3.mark_points(image.copy(), p0_tuples))

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters#answer-20805153

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            if y > int(image.shape[0]/2):
                # Assuming face start at the upper half of the image - TODO - is this a valid assumption?
                continue

            # Mark face area
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            # Find all facial features:
            features_in_face = [(p0_1d[i], p0_1d[i + 1]) for i in range(0, p0_1d.size, 2) if is_point_in_rect(x, y, w, h, p0_1d[i],p0_1d[i + 1])]

            # Extract relevant facial features
            # Eyes
            features_in_face_filtered = list()
            eyes = eyeCascade.detectMultiScale(roi_gray,scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))

            cnt = 0
            eye_bottom = list()
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                # Take center of bounding box - works better than feature filtering
                eye_feature = (x + ex + int(ew/2), y + ey + int(eh/2))
                # Left eye always first in list
                if len(features_in_face_filtered) > 0:
                    if ex < features_in_face_filtered[0][0]:
                        features_in_face_filtered.insert(0, eye_feature)
                    else:
                        features_in_face_filtered.append(eye_feature)
                else:
                    features_in_face_filtered.append(eye_feature)

                # features_in_eyes = [feature for feature in features_in_face if is_point_in_rect(x+ex, y+ey, ew, eh, feature[0],feature[1])]
                # if len(features_in_eyes) > 0:
                #     features_in_face_filtered.append(features_in_eyes[0])
                eye_bottom.append(ey+eh)
                # Detect only 2 eyes
                cnt += 1

            eye_bottom_avg = sum(eye_bottom)/len(eye_bottom) # With respect to top of roi
            # Filter out eyes that are far from average


            # Nose
            nose = noseCascade.detectMultiScale(roi_gray) #, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            for (nx, ny, nw, nh) in nose:
                if ny - eye_bottom_avg > 5 and len(nose) > 1:
                    # Assumption that nose upper border is very close to eye bottom border
                    continue
                cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
                # Take center of bounding box - works better than feature filtering
                nose_feature = (x + nx + int(nw/2), y + ny + int(nh/2))
                features_in_face_filtered.append(nose_feature)
                # # Take nose from  detected features
                # features_in_nose = [feature for feature in features_in_face if is_point_in_rect(x+nx, y+ny, nw, nh, feature[0],feature[1])]
                # if len(features_in_nose) > 0:
                #     features_in_face_filtered.append(features_in_nose[0])
                break

            if (len(nose) == 0 or len(features_in_face_filtered) < 3):
                # If no nose was detected, take nose coordinates of last nose
                print("Nose wasn't detected in frame " + str(i) + " - taking last detected nose point")
                features_in_face_filtered.append(nose_feature)
                no_nose_cnt += 1

            # Mouth
            mouth = mouthCascade.detectMultiScale(roi_gray) #
            # for (mx, my, mw, mh) in mouth:
            #     if my < int(h/2):
            #         # Mouth - Assuming in lower half of mouth to filter false detections - TODO - is this a valid assumption?
            #         continue
            #     cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            #     features_in_mouth = [feature for feature in features_in_face if is_point_in_rect(x+mx, y+my, mw, mh, feature[0],feature[1])]
            #     if len(features_in_mouth) > 0:
            #         features_in_face_filtered.append(features_in_mouth[0])
            #     break

            # Plot filtered features on frame
            # utils.cvshow("goodFeaturesToTrack", q3.mark_points(image, features_in_face))
            q3.mark_points(image, features_in_face_filtered)

            if len(features_in_face_filtered) < 3:
                print("Found {0} faces, {1} eyes, {2} nose and {3} mouth in and total of {4} facial features in frame {5}! ".format(len(faces), len(eyes), len(nose), len(mouth), len(features_in_face_filtered), i))
                # if len(features_in_face_filtered) < 3 :
                #     very_bad_frames_cnt += 1
                # else:
                #     bad_frames_cnt += 1

        cv2.imwrite(haar_dir_results + '/' + str(i) + '.jpg', image)
        # cv2.imshow("Faces found", image)
        # cv2.waitKey()
        frames_features.append(features_in_face_filtered)
    print("Nose wasn't detected in " + str(no_nose_cnt) + " frames")
    print("number of bad faces is: " + str(bad_frames_cnt) + " and very bad faces is: " + str(very_bad_frames_cnt))
    return frames_features


if __name__ == "__main__":
    features = detect_features()


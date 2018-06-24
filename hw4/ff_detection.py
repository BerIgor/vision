import cv2
from hw4 import utils


haar_dir = utils.get_pwd() + "/haar_xmls"


def detect_face_features(image):
    faces = detect_face_features(image)

    for (x, y, w, h) in faces:
        eyes = detect_eyes()
















def detect_faces(image):
    faceCascPath = haar_dir + "/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(faceCascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    filtered_faces = list()
    for (x, y, w, h) in faces:
        if y > int(image.shape[0] / 2):
            # Assuming face start at the upper half of the image - TODO - is this a valid assumption?
            continue
        filtered_faces.append((x, y, w, h))

    return filtered_faces


def detect_eyes(image, roi):
    eyeCascPath = haar_dir + "/haarcascade_eye.xml"
    eyeCascade = cv2.CascadeClassifier(eyeCascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))



def detect_noses(image, roi):
    noseCascPath = haar_dir + "/Nariz_nose.xml"
    noseCascade = cv2.CascadeClassifier(noseCascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nose = noseCascade.detectMultiScale(roi_gray)


def detecet_mouths(image, roi):
    mouthCascPath = haar_dir + "/Mouth.xml"
    mouthCascade = cv2.CascadeClassifier(mouthCascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mouth = mouthCascade.detectMultiScale(roi)



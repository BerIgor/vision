import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import torchvision as tv
# import torchvision.transforms as transforms
# from torch.autograd import Variable
from PIL import Image

import sys, os
import torch

# Prepare paths
pwd = os.getcwd().replace('\\','//')
q3 = pwd + '/q3'

sys.path.append(pwd + '/q3')
sys.path.append(pwd + '/q3/pytorch_segmentation_detection/')
sys.path.insert(0, pwd + '/q3/vision/')

from torchvision import transforms
from torch.autograd import Variable
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated


model_path = pwd + '/q3/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_34_8s_68.pth'
source_video_path = pwd + '/our_data/ariel.mp4'
target_video_path = pwd + '/our_data/ariel_m.avi' # OpenCV must have avi as output. https://github.com/ContinuumIO/anaconda-issues/issues/223#issuecomment-285523938

def image_get_fg_mask(image):
    fcn = resnet_dilated.Resnet34_8s(num_classes=21)
    fcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    fcn.eval()
    image = prep_image(image)
    nn_result = fcn(image)
    _, tmp = nn_result.squeeze(0).max(0)
    segmentation = tmp.data.cpu().numpy().squeeze()
    mask = np.zeros(np.shape(segmentation))
    # note: there can be more values, but 0 is bg
    mask[segmentation == 15] = 1
    return mask


def create_masked_video(src_video, trgt_video):
    video_reader = cv.VideoCapture(src_video)
    more_frames = True
    more_frames, frame = video_reader.read()
    rows = frame.shape[0]
    cols = frame.shape[1]

    video_format = cv.VideoWriter_fourcc(*"XVID")
    video_writer = cv.VideoWriter(trgt_video, video_format, 10, (cols,rows)) # In the constructor (column, row). However in video_writer.write its (row, column).

    i = 0
    while more_frames:
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        more_frames, frame = video_reader.read()
        if not more_frames:
            break

        # Correcting frame rotation
        rot_mat = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 270, 1)
        frame = cv.warpAffine(frame, rot_mat, (cols, rows))
        # Segment and remove background from video
        mask = image_get_fg_mask(frame)
        masked = apply_mask(frame, mask)
        # Write segmented image to output video
        video_writer.write(masked)
        print(i)
        i += 1

    video_writer.release()
    print("DONE")
    return


def video_to_frames(video_name):
    video = cv.VideoCapture(video_name)
    while video.isOpened():
        ret, frame = video.read()
        print(np.shape(frame))
        # print(ret)
        break
        # if key == ord('q'):
        #    break
    return frame


def prep_image(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    timage = transform(image)
    timage = timage.unsqueeze(0)
    return Variable(timage)


def cvshow(title, im):
    cv.namedWindow(title, cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL)
    cv.imshow(title, im)
    cv.waitKey()


def apply_mask(image, mask):
    for i in range(3):
        layer = image[:, :, i]
        layer[mask == 0] = 0
        image[:, :, i] = layer
    return image


if __name__ == "__main__":
    print("Welcome to q3")
    # frame = video_to_frames(video_path)
    # frame = cv.imread('our_data/GAzE2.jpeg')
    # nn_test(frame)
    # mask = image_get_fg_mask(frame)
    # cvshow("mask", mask)
    # masked = apply_mask(frame, mask)
    create_masked_video(source_video_path, target_video_path)
    # cvshow("after mask applied", masked)


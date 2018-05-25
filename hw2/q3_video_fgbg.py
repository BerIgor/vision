import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import torchvision as tv
# import torchvision.transforms as transforms
# from torch.autograd import Variable
from PIL import Image

import sys, os
import torch
sys.path.append('C:/GitProjects/vision/hw2/q3')
sys.path.append('C:/GitProjects/vision/hw2/q3/pytorch_segmentation_detection/')
sys.path.insert(0, 'C:/GitProjects/vision/hw2/q3/vision/')


from torchvision import transforms
from torch.autograd import Variable
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated


model_path = 'C:/GitProjects/vision/hw2/q3/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_34_8s_68.pth'
video_path = 'C:/GitProjects/vision/hw2/our_data/ariel.mp4'


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


def video_to_frames(video_name):
    video = cv.VideoCapture(video_name)
    while True:
        ret, frame = video.read()
        # cvshow("TEST", frame)
        # key = cv.waitKey(1000)
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
    frame = cv.imread('our_data/GAzE2.jpeg')
    # nn_test(frame)
    mask = image_get_fg_mask(frame)
    cvshow("mask", mask)
    masked = apply_mask(frame, mask)
    cvshow("after mask applied", masked)
    # image_get_fg_mask(frame)


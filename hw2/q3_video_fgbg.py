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
video_path = pwd + '/our_data/ariel.mp4'


def image_get_fg_mask(image):
    fcn = resnet_dilated.Resnet34_8s(num_classes=21)
    fcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    fcn.eval()
    image = prep_image(image)
    nn_result = fcn(image)
    _, tmp = nn_result.squeeze(0).max(0)
    segmentation = tmp.data.cpu().numpy().squeeze()
    print(np.shape(segmentation))
    mask = np.zeros(np.shape(segmentation))
    mask[segmentation == 15] = 1
    plt.imshow(segmentation)
    plt.show()
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
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.multiply(image, mask_rgb)
    return masked_image


if __name__ == "__main__":
    print("Welcome to q3")
    # frame = video_to_frames(video_path)
    frame = cv.imread('our_data/GAzE2.jpeg')
    # nn_test(frame)
    mask = image_get_fg_mask(frame)
    masked = apply_mask(frame, mask)
    cvshow("mask", masked)
    print("Finished")
    # image_get_fg_mask(frame)


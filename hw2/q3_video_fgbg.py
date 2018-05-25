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


def image_get_fg_mask(image):
    fcn = resnet_dilated.Resnet34_8s(num_classes=21)
    fcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    fcn.eval()
    cvshow("orig", image)
    image = prep_image(image)
    nn_result = fcn(image)
    cvshow(image)
    _, tmp = nn_result.squeeze(0).max(0)
    segmentation = tmp.data.cpu().numpy().squeeze()

    plt.imshow(segmentation)
    plt.show()


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


def display_images(image_list):
    f, sps = plt.subplots(nrows=1, ncols=len(image_list))
    # plt.gray()

    for i in range(0, len(image_list)):
        sps[i].imshow(image_list[i])
        sps[i].axis('off')

    plt.show()
    return


def cvshow(title, im):
    cv.namedWindow(title, cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL)
    cv.imshow(title, im)
    cv.waitKey()


if __name__ == "__main__":
    print("Welcome to q3")
    video_fname = 'our_data/cup.mov'
    frame = video_to_frames(video_fname)
    # nn_test(frame)
    image_get_fg_mask(frame)
    print("Finished")
    # image_get_fg_mask(frame)


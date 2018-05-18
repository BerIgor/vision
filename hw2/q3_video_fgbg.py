import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image


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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    our_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
    ])
    image = our_transform(image).float()
    image = image.unsqueeze(0)
    image = Variable(image, requires_grad=True)
    return image


def nn_test(image):
    pil_image = Image.fromarray(image)
    pil_image = prep_image(pil_image)
    dn = tv.models.resnet101(pretrained=True)
    dn.eval()
    # print(image_tensor)
    result = dn(pil_image)
    print(result)


# 0 is black, 255 is white
def image_get_fg_mask_me(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print("Welcome image_get_fg_mask")

    # _, thresh = cv.threshold(image_gray, 80, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    _, thresh = cv.threshold(image_gray, 80, 255, cv.THRESH_BINARY_INV)
    # thresh contains a mask with 0 as the object, and 255.0 as the bg (but some marked as bg are actually fg)
    cvshow("thresh", thresh)

    # Use open to remove noise
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    cvshow("openning", opening)

    # opening should be doing this on its own, the fuck
    # Return to original size
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sure_bg = 255 - cv.dilate(opening, kernel, iterations=3)
    cvshow("sure_bg", sure_bg)
    # BEFORE INVERSION sure_bg is a mask where 255.0 is the fg and 0.0 is the bg

    # distance transform assigns gray scale values based on the distance of each foreground (255) area from closest 0
    # color inversion should be considered as it might work faster

    dist_transform = cv.distanceTransform(sure_bg, cv.DIST_L2, 5)
    cvshow("dist", dist_transform)

    # Run thresholding to say that only values at certain distance are the fg/bg
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    cvshow("unknown", unknown)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    cvshow("CC", markers)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    cv.destroyAllWindows()
    cvshow("m1", markers)
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers[sure_bg == 255] = 0

    markers = cv.watershed(image, markers)
    cvshow("markers", markers)
    image[markers == -1] = [255, 0, 0]
    cvshow("END", image)

    return


def image_get_fg_mask(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers)
    print(np.unique(markers))

    img[markers == -1] = [255, 255, 255]

    cvshow("END", img)


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
    video_fname = 'our_imgs/banana.mov'
    frame = video_to_frames(video_fname)
    nn_test(frame)
    print("Finished")
    # image_get_fg_mask(frame)


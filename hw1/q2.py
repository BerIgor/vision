from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import filters as nfilters
from skimage import feature
from skimage import filters
import numpy as np
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
from sklearn.preprocessing import Binarizer


# Returns dic H(image_name)=image
def load_images(postfix="", exten="jpg"):
    _images = dict()
    _image_names = ["Church", "Golf", "Nuns"]
    image_dir = "edges_images_GT/"
    for image_name in _image_names:
        _images[image_name] = ndimage.imread(image_dir + image_name + postfix + "." + exten)
    return _images


# Returns a sobel filtered image
def run_sobel(image, thresh=0.0):
    filtered = filters.sobel(image)
    filtered = apply_threshold(filtered, thresh)
    return filtered


# Returns a canny filtered image
def run_canny(image, sigma=1, thresh=0.0):
    filtered = feature.canny(image, sigma=sigma)
    filtered = apply_threshold(filtered, thresh)
    return filtered


def apply_threshold(image, threshold=0.0):
    image = image > threshold
    return image


def edge_detection(name_to_image):
    for name, _ in name_to_image.items():
        image_o = name_to_image[name]

        fig, (sb0, sb1, sb2) = plt.subplots(nrows=1, ncols=3)
        plt.gray()

        sb0.imshow(image_o)
        sb0.axis('off')
        sb0.set_title("Original")

        sb1.imshow(run_canny(image_o, sigma=3, thresh=0.0))
        sb1.axis('off')
        sb1.set_title("Canny")

        sb2.imshow(run_sobel(image_o, thresh=0.0))
        sb2.axis('off')
        sb2.set_title("Sobel")

        plt.show()
    return


# The time complexity is high, but we don't mind
def calc_f_measure(name_to_image, name_to_image_gt):
    thresh_range = np.arange(0.0, 1.0, 0.01)
    binarizer = Binarizer(threshold=0.5)
    for thresh in thresh_range:
        # total_precision = 0
        # total_recall = 0
        total_f_measure = 0
        for name, _ in name_to_image.items():
            image = name_to_image[name]
            gt_image = name_to_image_gt[name]
            filtered_image = run_sobel(image)
            prediction = apply_threshold(filtered_image, thresh)

            # Data standardization
            gt_image = binarizer.transform(gt_image)
            prediction = binarizer.transform(prediction)

            current_f_measure = metrics.f1_score(gt_image, prediction, average='micro')
            total_f_measure += current_f_measure
            # current_precision = metrics.precision_score(gt_image, prediction, average='micro')
            # current_recall = metrics.recall_score(gt_image, prediction, average='micro')
            # print(str(current_precision) + str(current_recall))
            # total_precision += current_precision
            # total_recall += current_recall

        # avg_precision = total_precision / 3
        # avg_recall = total_recall / 3
        f_measure = total_f_measure / 3
        print(f_measure)

    return


if __name__ == "__main__":
    name_to_image_ = load_images()
    name_to_image_gt_ = load_images(postfix="_GT", exten="bmp")
    # edge_detection(name_to_image_)
    calc_f_measure(name_to_image_, name_to_image_gt_)

    # gt = np.array(([1.0, 1.0], [0.0, 1.0]))
    # binarizer = Binarizer(threshold=0.5)
    # print(gt)
    # gtn = binarizer.transform(gt)
    # print(gtn)
    # print(type(gt))
    # print(gt)
    # print(str(metrics.recall_score(gt, gtn, average='micro')))


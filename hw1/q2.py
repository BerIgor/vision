from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import feature
from skimage import filters
import numpy as np
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
    filtered = filtered > thresh
    return filtered


# Returns a canny filtered image
def run_canny(image, sigma=1, thresh=0.0):
    filtered = feature.canny(image, sigma=sigma, low_threshold=thresh, high_threshold=thresh )
    return filtered


def edge_detection(name_to_image):
    for name, _ in name_to_image.items():
        image_o = name_to_image[name]

        fig, sps = plt.subplots(nrows=3, ncols=2)
        plt.gray()

        sps[0, 0].imshow(image_o)
        sps[0, 0].axis('off')
        sps[0, 0].set_title("Original")

        sps[0, 1].imshow(run_canny(image_o, sigma=1.5, thresh=None))
        sps[0, 1].axis('off')
        sps[0, 1].set_title("Canny, σ = 1.5, Default Threshold")

        sps[1, 1].imshow(run_canny(image_o, sigma=1.5, thresh=0.5*255))
        sps[1, 1].axis('off')
        sps[1, 1].set_title("Canny, σ = 1.5, threshold = 0.5")

        sps[2, 1].imshow(run_canny(image_o, sigma=4, thresh=None))
        sps[2, 1].axis('off')
        sps[2, 1].set_title("Canny, σ = 4, Default Threshold")

        sps[1, 0].imshow(run_sobel(image_o, thresh=0.1))
        sps[1, 0].axis('off')
        sps[1, 0].set_title("Sobel, Threshold = 0.1")

        sps[2, 0].imshow(run_sobel(image_o, thresh=0.3))
        sps[2, 0].axis('off')
        sps[2, 0].set_title("Sobel, Threshold = 0.3")

        plt.show()
    return


# The time complexity is high, but we don't mind
def calc_f_measure(name_to_image, name_to_image_gt):
    thresh_range = np.arange(0.0, 0.5, 0.01)
    i = 0
    f_measure_vec = np.zeros((2, thresh_range.shape[0]))
    for thresh in thresh_range:
        total_f_measure_canny = 0
        total_f_measure_sobel = 0
        for name, _ in name_to_image.items():
            image = name_to_image[name]
            gt_image = name_to_image_gt[name]

            filtered_image_canny = run_canny(image, sigma=2, thresh=thresh*255)
            filtered_image_sobel = run_sobel(image, thresh=thresh)

            total_f_measure_canny += metrics.f1_score(gt_image, filtered_image_canny, average='micro')
            total_f_measure_sobel += metrics.f1_score(gt_image, filtered_image_sobel, average='micro')

        f_measure_vec[0,i] = total_f_measure_canny / 3
        f_measure_vec[1,i] = total_f_measure_sobel / 3
        i += 1
    print(f_measure_vec)
    plt.plot(thresh_range, f_measure_vec[0,:],label='Canny',color='red')
    plt.plot(thresh_range, f_measure_vec[1,:],label='Sobel',color='blue')
    plt.suptitle("F-measure With Respect to Threshold Value")
    plt.xlabel("Threshold")
    plt.ylabel("F-measure")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    name_to_image_ = load_images()
    name_to_image_gt_ = load_images(postfix="_GT", exten="bmp")
    edge_detection(name_to_image_)
    calc_f_measure(name_to_image_, name_to_image_gt_)

    r = np.array([False, True])
    gt = np.array([1.0, 1.0])
    f = metrics.f1_score(gt, r, average='micro')
    print(f)

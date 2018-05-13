import matplotlib.pyplot as plt
# import scipy
from scipy import ndimage
import scipy.ndimage.filters as filters
import cv2 as cv
from skimage.transform import pyramid_gaussian


def display_images(image_list):
    f, sps = plt.subplots(nrows=1, ncols=len(image_list))
    plt.gray()

    for i in range(0, len(image_list)):
        sps[i].imshow(image_list[i], cmap=plt.cm.gray)

    plt.show()
    return


def laplacian_decompose(image, levels):
    f_images = list()

    previous_filtered = image
    previous_sigma = 2
    for i in range(0, levels-1):
        print(i)
        current_sigma = previous_sigma * 2
        # current_filtered = filters.gaussian_filter(image, current_sigma, mode='wrap')
        kernel_size = 10*current_sigma + 1
        current_filtered = cv.GaussianBlur(src=image, ksize=(kernel_size, kernel_size), sigmaX=current_sigma)
        print("curr sigma == " + str(current_sigma))
        sub_res = previous_filtered - current_filtered
        cv.namedWindow("TEST", cv.WINDOW_AUTOSIZE)
        cv.imshow("TEST", sub_res, )
        cv.waitKey()
        '''
        plt.gray()
        plt.figure()
        plt.imshow(sub_res)
        plt.show()
        '''
        f_images.append(sub_res)
        previous_sigma = current_sigma
        previous_filtered = current_filtered
    f_images.append(filters.gaussian_filter(image, previous_sigma * 2))

    return f_images


if __name__ == "__main__":
    # image = ndimage.imread(fname="our_imgs/vettel.jpg", mode="L")
    image = cv.imread("our_imgs/cameraman.jpg")
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gauss_images = laplacian_decompose(image, 5)
    pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True))
    print('end')
    # display_images(pyramid)


'''
fig, sps = plt.subplots(nrows=3, ncols=2)
plt.gray()

sps[0, 0].imshow(image_o)
sps[0, 0].axis('off')
sps[0, 0].set_title("Original")

sps[0, 1].imshow(run_canny(image_o, sigma=1.5, thresh=None))
sps[0, 1].axis('off')
sps[0, 1].set_title("Canny, σ = 1.5, Default Threshold")
'''

# Comment to push - verify git is working
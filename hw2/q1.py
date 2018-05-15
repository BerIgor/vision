import numpy as np
import matplotlib.pyplot as plt
# import scipy
from scipy import ndimage
import scipy.ndimage.filters as filters
import cv2 as cv
import imageio as imio


def display_images(image_list):
    f, sps = plt.subplots(nrows=1, ncols=len(image_list))
    plt.gray()

    for i in range(0, len(image_list)):
        sps[i].imshow(image_list[i])

    plt.show()
    return


def laplacian_decompose(image, levels):
    f_images = list()

    previous_filtered = image
    previous_sigma = 2
    kernel_size = 21
    for i in range(0, levels-1):
        print(i)
        current_sigma = previous_sigma * 2
        current_filtered = filters.gaussian_filter(image, current_sigma)
        #kernel_size = 5*current_sigma + 1
        #current_filtered = cv.GaussianBlur(src=image, ksize=(kernel_size, kernel_size), sigmaX=current_sigma)
        print("curr sigma == " + str(current_sigma))
        sub_res = previous_filtered - current_filtered
        # cvshow("TEST", sub_res)

        f_images.append(sub_res)
        previous_sigma = current_sigma
        previous_filtered = current_filtered
    f_images.append(previous_filtered)

    return f_images


def reconstruct_from_laplcian_piramid(lpyramid):
    m_tot = 0
    for m in lpyramid:
        m_tot += m
    return m_tot


def cvshow(title, im):
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.imshow(title, im)
    cv.waitKey()


def part1():
    image = ndimage.imread(fname='data/Inputs/imgs/0006_001.png', mode="L")
    # image = cv.imread('data/Inputs/imgs/0006_001.png')
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # For our own RGB input
    # cv.imshow("original", image)
    laplacian_pyramid = laplacian_decompose(image, 5)
    # pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True)) # needs more imports
    print('end')
    # display_images(laplacian_pyramid)
    reconstruction = reconstruct_from_laplcian_piramid(laplacian_pyramid)
    cvshow("Reconstruction", reconstruction)

################################################################################
#                                PART 2
################################################################################
def part2():
    input_image = p2_prep_image('data/Inputs/imgs/0004_6.png')
    example_image = p2_prep_image('data/Examples/imgs/6.png')
    change_background()


def p2_prep_image(imname):
    image = ndimage.imread(fname=imname, mode="L")
    image_n = np.zeros(np.shape(image))
    cv.normalize(image.astype("double"), dst=image_n, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
    return image_n


def change_background():
    # trash
    iimage_mask = p2_prep_image('data/Inputs/masks/0004_6.png')
    eimage_bg = p2_prep_image('data/Examples/bgs/6.jpg')

    input_image = p2_prep_image('data/Inputs/imgs/0004_6.png')
    example_image = p2_prep_image('data/Examples/imgs/6.png')

    # end of trash
    # result = np.zeros(np.shape(input_image))
    # cvshow("first", np.multiply(eimage_bg, iimage_mask))
    # cvshow("second", np.multiply(input_image, 1-iimage_mask))
    a1 = np.multiply(eimage_bg, iimage_mask)
    a2 = np.multiply(input_image, 1 - iimage_mask)

    result = np.multiply(eimage_bg, iimage_mask) + np.multiply(input_image, 1-iimage_mask)
    cvshow("gogo", result)



if __name__ == "__main__":
    # # # part 1 # # #
    # part1()

    # # # part 2 # # #
    part2()

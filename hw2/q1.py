import numpy as np
import matplotlib.pyplot as plt
# import scipy
from scipy import ndimage
import scipy.ndimage.filters as filters
import cv2 as cv
import imageio as imio
import scipy.misc as misc


def display_images(image_list):
    f, sps = plt.subplots(nrows=1, ncols=len(image_list))
    plt.gray()

    for i in range(0, len(image_list)):
        sps[i].imshow(image_list[i])
        sps[i].axis('off')

    plt.show()
    return


def laplacian_decompose(image, levels):
    f_images = list()

    previous_filtered = image
    previous_sigma = 2
    kernel_size = 21
    for i in range(0, levels-1):
        current_sigma = previous_sigma * 2
        current_filtered = filters.gaussian_filter(image, current_sigma)
        #kernel_size = 5*current_sigma + 1
        #current_filtered = cv.GaussianBlur(src=image, ksize=(kernel_size, kernel_size), sigmaX=current_sigma)
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
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, im)
    cv.waitKey()


def part1():
    image = ndimage.imread(fname='data/Inputs/imgs/0006_001.png', mode="L")
    # image = cv.imread('data/Inputs/imgs/0006_001.png')
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # For our own RGB input
    # cv.imshow("original", image)
    laplacian_pyramid = laplacian_decompose(image, 5)
    # pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True)) # needs more imports
    # display_images(laplacian_pyramid)
    reconstruction = reconstruct_from_laplcian_piramid(laplacian_pyramid)
    cvshow("Reconstruction", reconstruction)


################################################################################
#                                PART 2
################################################################################
def part2():
    input_image, example_image, bg_image, mask = p21_read_images("0004_6", "6")
    input_w_bg = p21_apply_background(input_image, bg_image, mask)

    output_channels = list()
    # p22
    for channel_num in range(0, np.shape(input_w_bg)[2]):
        print(channel_num)
        input_channel = input_w_bg[:, :, channel_num]
        example_channel = example_image[:, :, channel_num]

        input_laplacian_pyramid = laplacian_decompose(input_channel, 6)
        example_laplacian_pyramid = laplacian_decompose(example_channel, 6)
        output_laplacian_pyramid = list()
        for level in range(0, len(input_laplacian_pyramid)):
            gain = p22_calc_gain(input_laplacian_pyramid[level], example_laplacian_pyramid[level], level+1)
            output_laplacian_pyramid.append(np.multiply(gain, input_laplacian_pyramid[level]))

        output_channels.append(output_laplacian_pyramid)
    # now we have a list of channels, and per channel, we have a pyramid

    p22_reconstruct_image(output_channels)
    return


def p2_prep_image(file_name):
    image = ndimage.imread(fname=file_name, mode='RGB')
    image_n = image / 255
    return image_n


def p21_read_images(input_name, example_name):
    input_image = p2_prep_image('data/Inputs/imgs/' + str(input_name) + '.png')
    example_image = p2_prep_image('data/Examples/imgs/' + str(example_name) + '.png')
    mask = p2_prep_image('data/Inputs/masks/' + str(input_name) + '.png')
    bg = p2_prep_image('data/Examples/bgs/' + str(example_name) + '.jpg')
    return input_image, example_image, bg, mask


def p21_apply_background(input_image, background_image, mask):
    result = np.multiply(background_image, 1-mask) + np.multiply(input_image, mask)
    return result


def p22_calc_energy(lap_level_image, level):
    left = np.power(lap_level_image, 2)
    energy = filters.gaussian_filter(left, level)
    return energy


def p22_calc_gain(input_level, example_level, level):
    input_energy = p22_calc_energy(input_level, level)
    example_energy = p22_calc_energy(example_level, level)
    denominator = example_energy
    numerator = input_energy + 0.0004
    gain = np.sqrt(numerator/denominator)
    return gain


# The input is a list of channels, and of each channel is a list
def p22_reconstruct_image(channel_pyramid_list):
    height, width = np.shape(channel_pyramid_list[0][0])
    result = np.zeros([height, width, 3])
    i = 2
    for channel in channel_pyramid_list:
        reconstructed_channel = reconstruct_from_laplcian_piramid(channel)
        result[:, :, i] = reconstructed_channel
        i += 1
        i, _ = divmod(i, 3)
    iml = list()
    iml.append(result)
    iml.append(result)
    display_images(iml)


if __name__ == "__main__":
    # # # part 1 # # #
    # part1()

    # # # part 2 # # #
    part2()

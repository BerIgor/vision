import numpy as np
import scipy.ndimage.filters as filters
from hw4 import utils


def transfer(input_image, example_image):

    output_channels = list()
    for channel_num in range(0, np.shape(input_image)[2]):
        input_channel = input_image[:, :, channel_num]
        example_channel = example_image[:, :, channel_num]

        input_laplacian_pyramid = laplacian_decompose(input_channel, 6)
        example_laplacian_pyramid = laplacian_decompose(example_channel, 6)
        output_laplacian_pyramid = list()
        for level in range(0, len(input_laplacian_pyramid)):
            gain = p22_calc_gain(input_laplacian_pyramid[level], example_laplacian_pyramid[level], level+1)
            if level == len(input_laplacian_pyramid)-1:
                output_laplacian_pyramid.append(example_laplacian_pyramid[level])
            else:
                output_laplacian_pyramid.append(np.multiply(gain, input_laplacian_pyramid[level]))

        output_channels.append(output_laplacian_pyramid)
    # now we have a list of channels, and per channel, we have a pyramid

    combined = p22_reconstruct_image(output_channels)
    combined = np.clip(combined, a_min=0.0, a_max=1.0)
    return combined


def laplacian_decompose(image, levels):
    f_images = list()
    previous_filtered = image
    previous_sigma = 2
    for i in range(0, levels-1):
        current_sigma = previous_sigma * 2
        current_filtered = filters.gaussian_filter(image, current_sigma)
        sub_res = previous_filtered - current_filtered

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


def p21_apply_background(input_image, background_image, mask):
    result = np.multiply(background_image, 1-mask) + np.multiply(input_image, mask)
    return result


def p22_calc_energy(lap_level_image, next_level):
    left = np.power(lap_level_image, 2)
    energy = filters.gaussian_filter(left, 2**next_level)
    return energy


def p22_calc_gain(input_level, example_level, level):
    input_energy = p22_calc_energy(input_level, level)
    example_energy = p22_calc_energy(example_level, level)
    numerator = example_energy
    denominator = np.add(input_energy, 0.0001)
    gain = np.sqrt(np.divide(numerator, denominator))
    gain = np.clip(gain, a_min=0.9, a_max=2.8)
    return gain


# The input is a list of channels, and of each channel is a list
def p22_reconstruct_image(channel_pyramid_list):
    height, width = np.shape(channel_pyramid_list[0][0])
    result = np.zeros([height, width, 3])
    i = 0
    for channel in channel_pyramid_list:
        reconstructed_channel = reconstruct_from_laplcian_piramid(channel)
        result[:, :, i] = reconstructed_channel
        i += 1

    # result = p21_apply_background(result)
    return result

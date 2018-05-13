import matplotlib.pyplot as plt
# import scipy
from scipy import ndimage
import scipy.ndimage.filters as filters


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
    for i in range(0, levels-1):
        print(i)
        current_sigma = previous_sigma * 2
        current_filtered = filters.gaussian_filter(image, current_sigma, mode='wrap')
        # current_filtered = filters.gaussian_filter(previous_filtered, current_sigma, mode='wrap')
        print("curr sigma == " + str(current_sigma))
        sub_res = previous_filtered - current_filtered
        f_images.append(sub_res)
        previous_sigma = current_sigma
        previous_filtered = current_filtered
    f_images.append(filters.gaussian_filter(image, previous_sigma * 2))

    return f_images


if __name__ == "__main__":
    image = ndimage.imread(fname="our_imgs/vettel.jpg", mode="L")
    gauss_images = laplacian_decompose(image, 5)
    display_images(gauss_images)


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
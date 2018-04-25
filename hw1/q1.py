import torchvision as tv
import matplotlib.pyplot as plt
import cv2 as cv
import torchvision.models as models
import numpy as np
from PIL import Image
from torch.autograd import Variable
from PIL import ImageFilter
import PIL.ImageOps
import torch.nn as nn
import scipy.ndimage as ndimage
from scipy import spatial

transforms = tv.transforms


# Load the birds images and classify them
def sub2():
    print("=================================")
    print("Running sub2: Bird classification")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    # image = cv.imread('birds/bird_0.jpg')
    #TODO: Handle both images
    image = Image.open('birds/bird_1.jpg')
    image = prep_image(image)
    classify(vgg16, image)


# Load our image (ice cream) and classify it
def sub3():
    print("======================================")
    print("Running sub3: Ice-Cream classification")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    image = Image.open('our_images/ice_cream.jpg')
    image = prep_image(image)
    classify(vgg16, image)


# Perform transformations and classify our image
def sub4():
    print("==================================================")
    print("Running sub4: Transformed Ice-Cream classification")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()

    image = Image.open('our_images/ice_cream.jpg')
    print("---Filtered Image---")
    image_filt = image.filter(ImageFilter.BLUR)
    image_filt = prep_image(image_filt)
    classify(vgg16, image_filt)

    print("---Different color Image---")
    image_color = PIL.ImageOps.invert(image)
    image_color = prep_image(image_color)
    classify(vgg16, image_color)

    print("---Geometric transform Image---")
    image_geo = image.rotate(45)
    image_geo = prep_image(image_geo )
    classify(vgg16, image_geo)


def sub5():
    print("================")
    print("Accessing layers")
    vgg16 = models.vgg16(pretrained=True)
    for child in vgg16.children():
        layer = child[0]
        print(layer)
        break
    layer_np = layer.weight.data.numpy()
    filters = layer_np[0:2, :, :, :]

    fig = plt.figure()
    plt.title('1st Conv. layer filters')
    plt.axis('off')
    for i in range(2):
        filter_3d = filters[i, :, :, :]
        print(filter_3d)
        sb = fig.add_subplot(1, 2, i+1)
        sb.axis('off')
        sb.imshow(filter_3d)
    fig.show()
    plt.waitforbuttonpress()

    # Prepare images
    ice_cream = Image.open('our_images/ice_cream.jpg')
    image_filt = ice_cream.filter(ImageFilter.BLUR)
    image_filt = prep_image(image_filt)
    image_filt = np.moveaxis(np.squeeze(image_filt.data.numpy()), 0, -1)
    image_color = PIL.ImageOps.invert(ice_cream)
    image_color = prep_image(image_color)
    image_color = np.moveaxis(np.squeeze(image_color.data.numpy()), 0, -1)
    image_geo = ice_cream.rotate(45)
    image_geo = prep_image(image_geo)
    image_geo = np.moveaxis(np.squeeze(image_geo.data.numpy()), 0, -1)
    image_list = list([image_filt, image_color, image_geo])

    fig2 = plt.figure()
    plt.title('1st Conv. layer response')
    plt.axis('off')
    subplot = 1
    for j in range(3):
        image = image_list[j]
        for i in range(2):
            response = ndimage.convolve(image, filters[i, :, :, :])
            print(str(i+j+1))
            sb = fig2.add_subplot(3, 2, subplot)
            sb.axis('off')
            sb.imshow(response)
            subplot += 1

    fig2.show()
    plt.waitforbuttonpress()
    return


def sub6():
    print("===================================================")
    print("Accessing features at FC7 (It's fucking FC2, not 7)")

    cats_dogs = get_images()

    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
    vgg16.eval()

    it = 0
    colors = ["#FF0000", 	"#800000"]

    for animal, images in cats_dogs.items():
        for image in images:
            result = vgg16(image)
            vector = np.squeeze(result.data.numpy())
            vector = [float('nan') if n < 1 else n for n in vector]
            xx = range(1, np.size(vector)+1)
            plt.scatter(xx, vector, s=0.1, c=colors[it])
        it += 1
    plt.show()


def sub7():
    print("===============================")
    print("Classifying our own cat and dog")
    cat_o = Image.open('our_images/cat.jpg')
    dog_o = Image.open('our_images/dog.jpg')
    cat = prep_image(cat_o)
    dog = prep_image(dog_o)

    fig = plt.figure()
    fig.suptitle("Our cat and dog")
    sp = fig.add_subplot(321)
    sp.imshow(cat_o)
    sp.axis('off')
    sp.set_title('Original cat')
    sp = fig.add_subplot(322)
    sp.imshow(dog_o)
    sp.axis('off')
    sp.set_title("Original dog")
    sp = fig.add_subplot(323)
    sp.imshow(np.moveaxis(np.squeeze(cat.data.numpy()), 0, -1))
    sp.axis('off')
    sp.set_title("NN ready cat")
    sp = fig.add_subplot(324)
    sp.imshow(np.moveaxis(np.squeeze(dog.data.numpy()), 0, -1))
    sp.axis('off')
    sp.set_title("NN ready dog")
    #plt.show()

    cats_dogs = get_images()

    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
    vgg16.eval()

    features = np.zeros([20, 4096])
    i = 0
    for animal, images in cats_dogs.items():
        # Note: the first animal handled is first animal in the array. This is not safe, but we don't mind
        for image in images:
            result = vgg16(image)
            vector = result.data.numpy()
            features[i, :] = vector
            i += 1

    tree = spatial.KDTree(features)
    cat_features = vgg16(cat).data.numpy()
    dist, pos = tree.query(cat_features, p=2)  # this is euclidian distance
    pos = pos[0]
    print("Image of cat -")
    if pos < 10:
        animal = "cats"
    else:
        animal = "dogs"
    pos = pos % 10

    print("Closest to " + str(animal) + " at " + str(pos))

    sp = fig.add_subplot(325)
    n_image = (cats_dogs[animal])[pos]
    sp.imshow(np.moveaxis(np.squeeze(n_image.data.numpy()), 0, -1))
    sp.set_title("Nearest neighbor")
    sp.axis('off')

    dog_features = vgg16(dog).data.numpy()
    dist, pos = tree.query(dog_features, p=2)
    pos = pos[0]
    print("Image of dog -")
    if pos < 10:
        animal = "cats"
    else:
        animal = "dogs"
    pos = pos % 10

    print("Closest to " + str(animal) + " at " + str(pos))

    sp = fig.add_subplot(326)
    n_image = (cats_dogs[animal])[pos]
    sp.imshow(np.moveaxis(np.squeeze(n_image.data.numpy()), 0, -1))
    sp.axis('off')
    sp.set_title("Nearest neighbor")
    plt.show()


def sub8():
    return


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


def classify(model, image):
    result = model(image)
    net_out = result.data.numpy()
    print(str(np.argmax(net_out)) + " " + str(np.max(net_out)))


# Returns a dict with all cat images under cats, and dog images under dogs
def get_images():
    # cats_dogs = list()
    cats_dogs = dict()
    cats = list()
    dogs = list()
    for i in range(10):
        dog = prep_image(Image.open("dogs\dog_" + str(i) + ".jpg"))
        cat = prep_image(Image.open("cats\cat_" + str(i) + ".jpg"))
        dogs.append(dog)
        cats.append(cat)
    # cats_dogs.append(cats)
    # cats_dogs.append(dogs)
    cats_dogs["cats"] = cats
    cats_dogs["dogs"] = dogs
    return cats_dogs


if __name__ == "__main__":
    # sub2()
    # sub3()
    # sub4()
    # sub5()
    # sub6()
    sub7()

import torchvision as tv
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
from PIL import Image
from torch.autograd import Variable
from PIL import ImageFilter
import PIL.ImageOps
import torch.nn as nn
import scipy.ndimage as ndimage
from scipy import spatial
from sklearn import svm


transforms = tv.transforms


# Load the birds images and classify them
def sub2():
    print("=============sub2================")
    print("Running sub2: Bird classification")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    bird_0_o = Image.open('birds/bird_0.jpg')
    bird_0 = prep_image(bird_0_o)
    bird_1_o = Image.open('birds/bird_1.jpg')
    bird_1 = prep_image(bird_1_o)

    bird_0_class, _ = classify(vgg16, bird_0)
    bird_1_class, _ = classify(vgg16, bird_1)

    fig = plt.figure()
    plt.suptitle("sub2 - Bird classification")
    sb = fig.add_subplot(221)
    sb.imshow(bird_0_o)
    sb.axis('off')
    sb.set_title("Original bird_0")
    sb = fig.add_subplot(222)
    sb.imshow(bird_1_o)
    sb.axis('off')
    sb.set_title("Original bird_1")

    sb = fig.add_subplot(223)
    sb.imshow(np.moveaxis(np.squeeze(bird_0.data.numpy()), 0, -1))
    sb.axis('off')
    sb.set_title("bird_0 - class == " + str(bird_0_class))
    sb = fig.add_subplot(224)
    sb.imshow(np.moveaxis(np.squeeze(bird_1.data.numpy()), 0, -1))
    sb.axis('off')
    sb.set_title("bird_1 - class == " + str(bird_1_class))
    plt.show()
    return


# Load our image (ice cream) and classify it
def sub3():
    print("===============sub3===================")
    print("Running sub3: Ice-Cream classification")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    image_o = Image.open('our_images/ice_cream.jpg')
    image = prep_image(image_o)
    classification, _ = classify(vgg16, image)

    fig = plt.figure()
    plt.suptitle("sub3 - Ice-Cream classification")
    sb = fig.add_subplot(121)
    sb.set_title("Original image")
    sb.axis('off')
    sb.imshow(image_o)

    sb = fig.add_subplot(122)
    sb.imshow(np.moveaxis(np.squeeze(image.data.numpy()), 0, -1))
    sb.set_title("NN ready image - classified as: " + str(classification))
    sb.axis('off')
    plt.show()
    return


# Perform transformations and classify our image
def sub4():
    print("=================sub4=============================")
    print("Running sub4: Transformed Ice-Cream classification")
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()

    image = Image.open('our_images/ice_cream.jpg')

    fig = plt.figure()
    plt.suptitle("sub4 - Classification under transformations")

    print("---Filtered Image---")
    image_filt_o = image.filter(ImageFilter.BLUR)
    image_filt = prep_image(image_filt_o)
    classification, _ = classify(vgg16, image_filt)
    sb = fig.add_subplot(131)
    sb.imshow(image_filt_o)
    sb.axis('off')
    sb.set_title("Blurred image - classified as: " + str(classification))

    print("---Different color Image---")
    image_color_o = PIL.ImageOps.invert(image)
    image_color = prep_image(image_color_o)
    classification, _ = classify(vgg16, image_color)
    sb = fig.add_subplot(132)
    sb.imshow(image_color_o)
    sb.axis('off')
    sb.set_title("Inverted color image - classified as: " + str(classification))

    print("---Geometric transform Image---")
    image_geo_o = image.rotate(45)
    image_geo = prep_image(image_geo_o)
    classification, _ = classify(vgg16, image_geo)
    sb = fig.add_subplot(133)
    sb.imshow(image_geo_o)
    sb.axis('off')
    sb.set_title("45deg. rotated image - classified as: " + str(classification))

    plt.show()
    return


def sub5():
    print("=====sub5=======")
    print("Accessing layers")
    vgg16 = models.vgg16(pretrained=True)
    for child in vgg16.children():
        layer = child[0]
        break
    layer_np = layer.weight.data.numpy()
    filters = layer_np[0:2, :, :, :]

    fig = plt.figure()
    plt.suptitle('sub5 - 1st Conv. layer filters')
    plt.axis('off')
    for i in range(2):
        filter_3d = filters[i, :, :, :]
        sb = fig.add_subplot(1, 2, i+1)
        sb.axis('off')
        sb.imshow(filter_3d)
    plt.show()

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
    plt.suptitle('sub5 - 1st Conv. layer response')
    plt.axis('off')
    subplot = 1
    for j in range(3):
        image = image_list[j]
        for i in range(2):
            response = ndimage.convolve(image, filters[i, :, :, :])
            sb = fig2.add_subplot(3, 2, subplot)
            sb.axis('off')
            sb.imshow(response)
            subplot += 1
    plt.show()
    return


def sub6():
    print("==========sub6===========")
    print("Accessing features at FC7")

    cats_dogs = get_images()

    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-3])
    vgg16.eval()

    it = 0
    colors = ["#FF0000", "#000000"]

    for animal, images in cats_dogs.items():
        for image in images:
            result = vgg16(image)
            vector = np.squeeze(result.data.numpy())
            xx = range(1, np.size(vector)+1)
            plt.scatter(xx, vector, s=0.1, c=colors[it])
        it += 1
    plt.suptitle("sub6 - Feature vectors extracted from FC7")
    plt.show()
    return


def sub7():
    print("=============sub7==============")
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

    cats_dogs = get_images()

    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-3])
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
    dist, pos = tree.query(cat_features, p=2)  # this is euclidean distance
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
    return


def sub8():
    print("==============sub8================")
    print("Classifying our own tiger and wolf")
    tiger_o = Image.open('our_images/tiger.jpg')
    wolf_o = Image.open('our_images/wolf.jpg')
    tiger = prep_image(tiger_o)
    wolf = prep_image(wolf_o)

    fig = plt.figure()
    fig.suptitle("Our tiger and wolf")
    sp = fig.add_subplot(321)
    sp.imshow(tiger_o)
    sp.axis('off')
    sp.set_title('Original tiger')
    sp = fig.add_subplot(322)
    sp.imshow(wolf_o)
    sp.axis('off')
    sp.set_title("Original wolf")
    sp = fig.add_subplot(323)
    sp.imshow(np.moveaxis(np.squeeze(tiger.data.numpy()), 0, -1))
    sp.axis('off')
    sp.set_title("NN ready tiger")
    sp = fig.add_subplot(324)
    sp.imshow(np.moveaxis(np.squeeze(wolf.data.numpy()), 0, -1))
    sp.axis('off')
    sp.set_title("NN ready wolf")

    cats_dogs = get_images()

    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-3])
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
    tiger_features = vgg16(tiger).data.numpy()
    dist, pos = tree.query(tiger_features, p=2)  # this is euclidian distance
    pos = pos[0]
    print("Image of tiger -")
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

    wolf_features = vgg16(wolf).data.numpy()
    dist, pos = tree.query(wolf_features, p=2)
    pos = pos[0]
    print("Image of wolf -")
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
    return


# Returns the SVM classifier created created in this function
# In this we cut off the network after the ReLU at FC7, because otherwise our tiger is a dog
def sub9():
    print("===========sub9============")
    print("Building our own classifier")

    cats_dogs = get_images()
    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
    vgg16.eval()

    svm_classifier = svm.SVC()

    features = np.zeros([20, 4096])
    tags = np.zeros([20])
    i = 0
    tag = 0  # 0 is cat, 1 is dog
    for animal, images in cats_dogs.items():
        # Note: the first animal handled is first animal in the array. This is not safe, but we don't mind
        for image in images:
            result = vgg16(image)
            vector = result.data.numpy()
            features[i, :] = vector
            tags[i] = tag
            i += 1
        tag += 1  # update the tag
    # Train SVM
    svm_classifier.fit(features, tags)

    # Obtain features
    cat_o = Image.open('our_images/cat.jpg')
    dog_o = Image.open('our_images/dog.jpg')
    cat = prep_image(cat_o)
    dog = prep_image(dog_o)
    cat_features = vgg16(cat).data.numpy()
    dog_features = vgg16(dog).data.numpy()

    # Classify two images
    result = svm_classifier.predict(cat_features)
    print("Cat classified as: " + str(result))
    result = svm_classifier.predict(dog_features)
    print("Dog classified as: " + str(result))
    print("Legend: 0-cat, 1-dog")
    return svm_classifier


def sub10(classifier):
    print("==================sub10=====================")
    print("Classifying our own tiger and wolf using SVM")
    tiger_o = Image.open('our_images/tiger.jpg')
    wolf_o = Image.open('our_images/wolf.jpg')
    tiger = prep_image(tiger_o)
    wolf = prep_image(wolf_o)

    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
    vgg16.eval()

    tiger_features = vgg16(tiger).data.numpy()
    result = classifier.predict(tiger_features)
    print("Tiger classified as: " + str(result))

    wolf_features = vgg16(wolf).data.numpy()
    result = classifier.predict(wolf_features)
    print("Wolf classified as: " + str(result))
    print("Legend: 0-cat, 1-dog")
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
    # print(str(np.argmax(net_out)) + " " + str(np.max(net_out)))
    return np.argmax(net_out), np.max(net_out)


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
    sub2()
    sub3()
    sub4()
    sub5()
    sub6()
    sub7()
    sub8()
    sub9_svm = sub9()
    sub10(sub9_svm)  # don't run without sub9

import cv2
import os

import numpy as np
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


canny_threshold1 = 50
canny_threshold2 = 150

IMG_H = 66
IMG_W = 200
IMG_C = 3
INPUT_SHAPE = (IMG_H, IMG_W, IMG_C)


def load_image(data_dir, image_file):
    """
    Load image
    :param data_dir:
    :param image_file:
    :return:
    """
    return mpimg.imread(os.path.join(data_dir, image_file))


def load_carla_image(image, path=None):
    """
    Load RGB image
    :param image:
    :param path:
    :return:
    """
    if path is not None:
        image = Image.open(os.path.join(path, image))
    else:
        image = Image.open(image)

    case1 = image.mode == 'RGBA'
    case2 = image.mode == 'CMYK'

    if case1 or case2:
        image = image.convert('RGB')

    image = img_to_array(image)
    image = image.astype(np.uint8)

    return image


def crop_img(image):
    """
    Crop Image
    :param image:
    :return:
    """
    return image[80:, :, :]


def resize_img(image):
    """
    Resize Image
    :param image:
    :return:
    """
    return cv2.resize(image, (IMG_W, IMG_H), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Change RGB to YUV
    :param image:
    :return:
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def random(center, left, right, steering_angle, offsets):
    """
    Choose a random image
    :param center:
    :param left:
    :param right:
    :param steering_angle:
    :param offsets:
    :return:
    """
    index = np.random.choice(3)
    if index == 0:
        return left, steering_angle + offsets

    elif index == 1:
        return right, steering_angle - offsets

    else:
        return center, steering_angle


def flip(image, steering_angle):
    """
    Flip image randomly
    :param image:
    :param steering_angle:
    :return:
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    else:
        image = image
        steering_angle = steering_angle

    return image, steering_angle


def translate(image, steering_angle, range_x, range_y):
    """
    Translate image randomly
    :param image:
    :param steering_angle:
    :param range_x:
    :param range_y:
    :return:
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, steering_angle


def shadow(image):
    """
    Create shadow
    :param image:
    :return:
    """
    a, b = np.random.rand(), np.random.rand()
    x1, y1 = IMG_W * a, 0
    x2, y2 = IMG_W * b, IMG_H

    xm, ym = np.mgrid[0:IMG_H, 0:IMG_W]

    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.5, high=1.0)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    return image


def adjust_darkness(image, a=0.5, b=0.5, c=0.6):
    """
    Adjust image darkness
    :param image:
    :param a:
    :param b:
    :param c:
    :return: image
    """
    x1, y1 = IMG_W * a, 0
    x2, y2 = IMG_W * b, IMG_H

    xm, ym = np.mgrid[0:IMG_H, 0:IMG_W]

    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    s_ratio = c
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1] = hls[:, :, 1] * s_ratio
    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    return image


def adjust_contrast(image, contrast=127):
    """
    Adjust image contrast
    :param image:
    :param contrast:
    :return: image
    """
    Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    Gamma = 127 * (1 - Alpha)

    # The function addWeighted calculates the weighted sum of two arrays
    image = cv2.addWeighted(image, Alpha, image, 0, Gamma)

    return image


def brightness(image):
    """
    Adjust brightness randomly
    :param image:
    :return:
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image


def transform_image(image):
    """
    Transform grayscale image from 1 channel to 3 channels
    :param image:
    :return:
    """
    image = np.stack((image,)*3, axis=-1)

    return image


def lane_image(image):
    """
    Detect edges in image
    :param image:
    :return:
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.Canny(image, canny_threshold1, canny_threshold2)
    image = transform_image(image)

    return image


def preprocess1(image):
    """
    preprocessing input
    :param image:
    :return:
    """
    image = adjust_darkness(image, c=0.75)
    image = adjust_contrast(image, contrast=20)
    image = crop_img(image)
    image = resize_img(image)
    image = rgb2yuv(image)
    image = adjust_darkness(image, c=0.75)
    image = adjust_contrast(image, contrast=40)

    return image


def preprocess2(image, path, steering_angle, range_x=100, range_y=10):
    """
    Augmenting image
    :param image:
    :param path:
    :param steering_angle:
    :param range_x:
    :param range_y:
    :return:
    """
    image = load_carla_image(image, path=path)
    image, steering_angle = flip(image, steering_angle)
    image, steering_angle = translate(image, steering_angle, range_x, range_y)

    return image, steering_angle


def data_generator(feature, label, path, batch_size, is_training, lucky_number=0.6):
    """
    Training generator
    :param feature:
    :param label:
    :param path:
    :param batch_size:
    :param is_training:
    :param lucky_number:
    :return:
    """
    images = np.empty([batch_size, IMG_H, IMG_W, IMG_C])
    angles = np.empty(batch_size)

    while True:
        i = 0
        for index in np.random.permutation(feature.shape[0]):
            image = feature[index]
            steering = label[index]

            if is_training and np.random.rand() < lucky_number:
                image, steering = preprocess2(image, path, steering)
            else:
                image = load_carla_image(image, path)

            images[i] = preprocess1(image)
            angles[i] = steering

            i += 1
            if i == batch_size:
                break

        yield images, angles

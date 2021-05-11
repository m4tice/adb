import os
import glob
import os
import keract
import cv2
import warnings

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from PIL import Image
from keract import get_activations
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


IMG_H = 66
IMG_W = 200
IMG_C = 3
INPUT_SHAPE = (IMG_H, IMG_W, IMG_C)

IM_WIDTH = 640
IM_HEIGHT = 480
LR = 1.0e-04


# - @private: Load RGB image
def load_image(image, limit, path=None):
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

    print("Carla image dtype   :", image.dtype)

    pimage = preprocess1(image, limit)
    pimage = pimage.reshape((1, pimage.shape[0], pimage.shape[1], pimage.shape[2]))

    return image, pimage


# @private: Crop Image
def crop_img(image, limit):
    # return image[60:-25, :, :]  # udacity
    return image[limit:, :, :]  # carla


# @private: Resize Image
def resize_img(image):
    return cv2.resize(image, (IMG_W, IMG_H), cv2.INTER_AREA)


# @private: Change RGB to YUV
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


# @public: [NEW]
def preprocess1(image, limit):
    image = crop_img(image, limit)
    image = resize_img(image)
    image = rgb2yuv(image)

    return image


def keract_process(image, model_name):
    # Load your model
    model = load_model(model_name)
    # model.summary()

    start_time = datetime.now()
    steer = model.predict(image)
    steer = float(steer[0][0])
    print("Predicted steering  :", steer)
    print("Predict time        :", datetime.now()-start_time)

    # activations = get_activations(model, image, auto_compile=True)
    # keract.display_activations(activations)
    # keract.display_heatmaps(activations, image, save=False)


def main():
    print("Tensorflow version  :", tf.version.VERSION)
    carla_model = "model_test.h5"
    print("Using model         :", carla_model)
    imgs = glob.glob("*.png")
    lims = [100, 300, 450, 100]

    for img, lim in zip(imgs, lims):
        print("\n{}".format("==" * 50))
        image, pimage = load_image(img, lim)
        print("Image WIDTH x HEIGHT: {} x {}".format(image.shape[1], image.shape[0]))
        start_time = datetime.now()
        _ = resize_img(image)
        print("Resize time         : {}".format(datetime.now() - start_time))
        keract_process(pimage, carla_model)

        cv2.imshow("", pimage[0])
        cv2.waitKey(0)

    plt.show()


if __name__ == '__main__':
    main()






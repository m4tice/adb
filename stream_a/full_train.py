import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

from sklearn.model_selection import train_test_split
from PIL import Image
from datetime import datetime
from time import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import img_to_array
from blockweek_ad.stream_a.nvidia_model import nvidia_model  # redirect to the correct model path if needed

import os
import cv2
import random

# -- PATH SETTINGS
"""
    Adjust training dataset here
"""
dataset = "dataset_01"

main_path = os.getcwd()
img_dir = os.path.join(main_path, dataset, "dataset")
csv_file = os.path.join(main_path, dataset, "dataset/driving_log.csv")

log_dir = os.path.join(main_path, dataset, "logs")
saved_logdir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
file_path = os.path.join(main_path, dataset, "models", "model-t4-{epoch:03d}-{val_loss:.6f}.h5")

# -- TRAINING PARAMETERS
"""
    Adjust training parameters here
"""
EPOCHS = 3
spe = 150
bs = 40
LR = 1.0e-04


def isfile(file):
    return os.path.isfile(file)


def isdir(path):
    return os.path.isdir(path)


print("-------- TRAINING PARAMETERS --------")
print("           IMG: ", isdir(img_dir))
print("      csv file: ", isfile(csv_file))
print("           Log: ", isdir(log_dir))
print("        EPOCHS: ", EPOCHS)
print("    Batch size: ", bs)
print(" Learning Rate: ", LR)
print("Step per epoch: ", spe)

# === DEFINE VARIABLES
IMG_H, IMG_W, IMG_C = 66, 200, 3
INPUT_SHAPE = (IMG_H, IMG_W, IMG_C)

# - training parameters ----
np.random.seed(0)

# - Lists
steering_list = []


# === FUNCTIONS
# - @private: Crop Image
def crop_img(image):
    return image[80:, :, :]


# - @private: Resize Image
def resize_img(image):
    return cv2.resize(image, (IMG_W, IMG_H), cv2.INTER_AREA)


# - @private: Change RGB to YUV
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


# - @private: adjust darkness
def adjust_darkness(image, a=0.5, b=0.5, c=0.6):
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


# - @private: adjust contrast
def adjust_contrast(image, contrast=127):
    Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    Gamma = 127 * (1 - Alpha)

    # The function addWeighted calculates the weighted sum of two arrays
    image = cv2.addWeighted(image, Alpha, image, 0, Gamma)

    return image


# - @private: Flip image randomly
def flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    else:
        image = image
        steering_angle = steering_angle

    return image, steering_angle


# - @private: Translate image randomly
def translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, steering_angle


# - @private: Load image
def load_image(image_path):
    return mpimg.imread(image_path)


# - @private: Load RGB image
def load_carla_image(image, path=None):
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


# - @public: [NEW]
def preprocess1(image):
    image = adjust_darkness(image, c=0.75)
    image = adjust_contrast(image, contrast=20)
    image = crop_img(image)
    image = resize_img(image)
    image = rgb2yuv(image)
    image = adjust_darkness(image, c=0.75)
    image = adjust_contrast(image, contrast=40)

    return image


# - @public: [NEW] Process normal image
def preprocess2(image, path, steering_angle, range_x=100, range_y=10):
    image = load_carla_image(image, path=path)  # carla
    image, steering_angle = flip(image, steering_angle)
    image, steering_angle = translate(image, steering_angle, range_x, range_y)

    return image, steering_angle


# - @public: [NEW] Data generator
def data_generator(feature, label, path, batch_size, is_training, lucky_number=0.5):
    images = np.empty([batch_size, IMG_H, IMG_W, IMG_C])
    angles = np.empty(batch_size)

    while True:
        i = 0
        for index in np.random.permutation(feature.shape[0]):
            image = feature[index]
            steering = label[index]

            if is_training and np.random.rand() < lucky_number:
                image, steering = preprocess2(image, path, steering, range_x=30)
            else:
                image = load_carla_image(image, path=path)  # carla

            images[i] = preprocess1(image)
            angles[i] = steering

            i += 1
            if i == batch_size:
                break

        yield images, angles


# - Loading CSV file
def load_data(amount=3):
    data_df = pd.read_csv(csv_file,
                          names=['center', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles',
                                 'braking', 'steering'])
    x = data_df['center'].values
    y = data_df['steering'].values
    y = np.ndarray.tolist(y)
    # print('>>> {} -- {} <<<'.format(len(x), len(y)))
    print('Data length: {}'.format(len(x)))

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)
    print(x_train[:amount])

    return x_train, x_valid, y_train, y_valid


def main():
    # load training data
    x_train, x_valid, y_train, y_valid = load_data()

    # load model
    model = nvidia_model()

    # Callbacks
    # -- ModelCheckpoint
    callback_modelcheckpoint = ModelCheckpoint(file_path,
                                               monitor='val_loss',
                                               verbose=0,
                                               save_best_only=False,
                                               mode='auto')

    # -- Tensorboard
    callback_tensorboard = tf.keras.callbacks.TensorBoard(saved_logdir, histogram_freq=1)

    # -- Final callbacks
    callbacks = [callback_modelcheckpoint, callback_tensorboard]

    # Compile
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LR))
    model.fit(data_generator(x_train, y_train, path=img_dir, batch_size=bs, is_training=True),
              steps_per_epoch=spe,
              epochs=EPOCHS,
              max_queue_size=1,
              validation_data=data_generator(x_valid, y_valid, path=img_dir, batch_size=bs, is_training=False),
              validation_steps=len(x_valid) / bs,
              callbacks=callbacks,
              verbose=1)


def test():
    x_train, x_valid, y_train, y_valid = load_data(amount=3)
    choice = random.randint(0, len(x_train) - 1)
    image = x_train[choice]
    steer = y_train[choice]
    print("IMG name :", image)
    print("Steer    :", type(steer))

    img = load_carla_image(image, path=img_dir)
    img = preprocess1(img)
    print("IMG dtype:", img.dtype)


if __name__ == '__main__':
    print("\n== BEFORE TRAINING ===================================================")
    test()
    print("\n== AFTER TRAINING ====================================================")
    main()

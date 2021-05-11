import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn.model_selection import train_test_split

import blockweek_ad.ca_utils.training_util as tu
from blockweek_ad.E2E.nvidia_model import nvidia_model


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
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


# - Loading CSV file
def load_data(amount=3):
    """
    Load training data
    :param amount:
    :return:
    """
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
    model.fit(tu.data_generator(x_train, y_train, path=img_dir, batch_size=bs, is_training=True),
              steps_per_epoch=spe,
              epochs=EPOCHS,
              max_queue_size=1,
              validation_data=tu.data_generator(x_valid, y_valid, path=img_dir, batch_size=bs, is_training=False),
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

    img = tu.load_carla_image(image, path=img_dir)
    img = tu.preprocess1(img)
    print("IMG dtype:", img.dtype)


if __name__ == '__main__':
    print("\n== BEFORE TRAINING ===================================================")
    test()
    print("\n== AFTER TRAINING ====================================================")
    main()

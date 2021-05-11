from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from blockweek_ad.ca_utils.training_util import INPUT_SHAPE


# - Build model
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

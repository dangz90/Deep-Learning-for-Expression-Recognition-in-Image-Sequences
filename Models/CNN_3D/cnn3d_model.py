import os
from keras.models import Sequential
from keras.utils.data_utils import get_file
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D

def get_model(summary=False, frames=16):
    WEIGHTS_PATH = 'sports1M_weights.h5'
    weights_path = os.path.join('weights', WEIGHTS_PATH)

    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', 
                            padding='same', name='conv1',
                            strides=(1, 1, 1), 
                            input_shape=(frames, 112, 112, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3), activation='relu', 
                            padding='same', name='conv2',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, (3, 3, 3), activation='relu', 
                            padding='same', name='conv3a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(256, (3, 3, 3), activation='relu', 
                            padding='same', name='conv3b',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 2, 2), 
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv4a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv4b',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 2, 2), 
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv5a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv5b',
                            strides=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=(2, 2, 2), 
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(12, activation='softmax', name='fc8'))
    if summary:
        print(model.summary())
    return model

model = get_model(summary=True)
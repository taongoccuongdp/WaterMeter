import os
from tensorflow.python import keras
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Bidirectional, Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.recurrent import LSTM
import tensorflow as tf
import numpy as np
"""
create rcnn model
parameters: - input_shape => tuple: shape of image input
            - num_classes => number of classes want to recognition 
return model
"""
def get_model(input_shape, num_classes):
    rcnn_model = Sequential()
    # using very-deep vgg to extract features
    rcnn_model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding=1, input_shape = input_shape, activation='relu'))
    rcnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    rcnn_model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding=1, activation='relu'))
    rcnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    rcnn_model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding=1, activation='relu'))
    rcnn_model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding=1, activation='relu'))
    rcnn_model.add(MaxPooling2D(pool_size=(1, 2), strides=2))
    rcnn_model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding=1, activation='relu'))
    rcnn_model.add(BatchNormalization())
    rcnn_model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding=1, activation='relu'))
    rcnn_model.add(BatchNormalization())
    rcnn_model.add(MaxPooling2D(pool_size=(1, 2), strides=2))
    rcnn_model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding=0, activation='relu'))
    rcnn_model.add(Flatten())
    #RNN with BLSTM
    rcnn_model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', dropout=0.25, recurrent_dropout=0.25)))
    rcnn_model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', dropout=0.25, recurrent_dropout=0.25)))
    rcnn_model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))
    return rcnn_model
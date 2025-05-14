import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
    #system
import os
import sys
import csv

import math
import random
import matplotlib
import matplotlib.pyplot as plt

def B_CNN_Model_B(input_shape, num_class_c, num_class_m, num_class_f, 
                  model_name:str='B_CNN_Model_B'):
    
    img_input = keras.layers.Input(shape=input_shape, name='input')

    #--- block 1 ---
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #--- block 2 ---
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #--- coarse 1 branch ---
    c_bch = keras.layers.Flatten(name='c1_flatten')(x)
    c_bch = keras.layers.Dense(256, activation='relu', name='c_fc_1')(c_bch)
    c_bch = keras.layers.BatchNormalization()(c_bch)
    c_bch = keras.layers.Dropout(0.5)(c_bch)
    c_bch = keras.layers.Dense(256, activation='relu', name='c_fc_2')(c_bch)
    c_bch = keras.layers.BatchNormalization()(c_bch)
    c_bch = keras.layers.Dropout(0.5)(c_bch)
    c_pred = keras.layers.Dense(num_class_c, activation='softmax', name='c_predictions')(c_bch)

    #--- block 3 ---
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #--- coarse 2 branch ---
    m_bch = keras.layers.Flatten(name='c2_flatten')(x)
    m_bch = keras.layers.Dense(512, activation='relu', name='m_fc_1')(m_bch)
    m_bch = keras.layers.BatchNormalization()(m_bch)
    m_bch = keras.layers.Dropout(0.5)(m_bch)
    m_bch = keras.layers.Dense(512, activation='relu', name='m_fc_2')(m_bch)
    m_bch = keras.layers.BatchNormalization()(m_bch)
    m_bch = keras.layers.Dropout(0.5)(m_bch)
    m_pred = keras.layers.Dense(num_class_m, activation='softmax', name='m_predictions')(m_bch)

    #--- block 4 ---
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #--- fine block ---
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(1024, activation='relu', name='f_fc_1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu', name='f_fc2_2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    f_pred = keras.layers.Dense(num_class_f, activation='softmax', name='f_predictions')(x)
    model = keras.Model(img_input, [c_pred, m_pred, f_pred], name=model_name)
    
    return model



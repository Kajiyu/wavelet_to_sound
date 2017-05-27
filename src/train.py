# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json, csv
import wave
from PIL import Image
from scipy import fromstring, int16

# keras系
from keras import models
from keras import layers
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense,Merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta
from keras.utils.generic_utils import Progbar
from keras.utils.visualize_util import plot
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


# Const Variable
N = 1378
IMG_SIZE = (137,275)
WAV_SIZE = 800

X = np.array([])
y = np.array([])


def set_model(model):
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(WAV_SIZE))
    adl = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.)
    model.compile(loss='categorical_crossentropy',optimizer=adl, metrics=['accuracy'])
    return model


def train():
    X = np.load('./data/train_X.npy')
    y = np.load('./data/train_y.npy')

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    train_generator = train_datagen.flow(X, y, batch_size=100)
    validation_generator = validation_datagen.flow(X,y)

    nb_epoch = 5
    nb_train_samples = 200
    nb_validation_samples = 100
    old_session = KTF.get_session()
    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)
        model = Sequential()
        with tf.name_scope("inference") as scope:
            model = set_model(model)
        model.summary()
        fpath = './model/weights.hdf5'
        tb_cb = TensorBoard(log_dir="./tensorlog", histogram_freq=1)
        cp_cb = ModelCheckpoint(filepath = fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        history = model.fit_generator(train_generator,samples_per_epoch=nb_train_samples, \
                                                        nb_epoch=nb_epoch, validation_data=validation_generator, \
                                                        nb_val_samples=nb_validation_samples,\
                                                        callbacks=[cp_cb, tb_cb])

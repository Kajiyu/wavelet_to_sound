# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json, csv
import wave
from PIL import Image
from scipy import fromstring, int16
import struct

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


def generate(img_paths, model):
    X = np.array([])
    i = 0
    for img_path in img_paths:
        print i
        image = Image.open(img_path).convert('RGB')
        im_arr = np.array(image)
        im_arr = im_arr.reshape((1,) + im_arr.shape)
        if i == 0:
            tmp_arr = im_arr
        elif  i == 1:
            X = np.vstack((tmp_arr, im_arr))
        else:
            X = np.vstack((X, im_arr))
        i = i + 1
    X = X.astype(np.float32) / 255.
    y = model.predict(X, batch_size=len(img_paths), verbose=0)
    y = (y - 0.5) * 2.0
    y = y * 32768.
    y = y.astype(np.int16)
    print y.shape
    for index in range(i):
        w = wave.Wave_write("./output/"+str(index)+".wav")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(11025)
        data = y[index]
        data = struct.pack("h" * len(data), *data)
        w.writeframes(data)
        w.close()
    w = wave.Wave_write("./connect.wav")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(11025)
    data = y.reshape(-1,)
    data = struct.pack("h" * len(data), *data)
    w.writeframes(data)
    w.close()

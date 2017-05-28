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
from src import train, generate, set_model
import glob

if __name__ == '__main__':
    # load model
    model = Sequential()
    model = set_model(model)
    model.load_weights('./model/weights.hdf5')

    img_paths = glob.glob('./imgs/*.png')
    generate(img_paths, model)

# %%

# -*- coding: utf-8 -*-
# Author : Ali Mirzaei
# Date : 19/09/2017


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Conv2DTranspose, concatenate
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import matplotlib
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import Image, display
import random
import math
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten, Activation, Concatenate, \
    LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, AvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend, models
# import tensorflow_addons as tfa
import tensorflow as tf

print(tf.__version__)

# need to add these for the GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# import the image generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%

# Setting the parameters for training

# batch size and image width to use
batch_size = 128
width = 100

# all the data directories
train_dir = 'train/';
test_dir = 'test/'
valid_dir = 'valid/';

# the number of epochs
num_epochs = 5000

# creating an image generator that will feed the data from
# each of the directories

# we use scaling transformation in this generator
generator = ImageDataGenerator(rescale=1. / 255)

# we specify the size of the input and batch size
# size of the input is necessary because the image
# needs to be rescaled for the neural network

train_data = generator.flow_from_directory(train_dir, target_size=(width, width), batch_size=10000)
valid_data = generator.flow_from_directory(valid_dir, target_size=(width, width), batch_size=1000)
test_data = generator.flow_from_directory(test_dir, target_size=(width, width), batch_size=1000)

# the number of steps per epoch is samples/batch size
# we need to use these numbers later

train_steps_per_epoch = math.ceil(train_data.samples / batch_size)
valid_steps_per_epoch = math.ceil(valid_data.samples / batch_size)
test_steps_per_epoch = math.ceil(test_data.samples / batch_size)


def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
    Argument:
        rgb (tensor): rgb image
    Return:
        (tensor): grayscale image
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


x_train, y_train = train_data.next()
x_test, y_test = test_data.next()
x_train.shape
num_labels = len(train_data.class_indices)

import matplotlib.pyplot as plt


def show_image(x):
    plt.imshow(np.clip(x, 0, 1))


# %%

show_image(x_train[1])

# %%


# %%

initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)
from keras import backend as K

epochs = 5000

img_rows = width
img_cols = width
channels = 3

# network parameters
input_shape = (img_rows, img_cols, channels)
batch_size = 128
kernel_size = 3
filters = 16
latent_dim = 256
# encoder/decoder number of CNN layers and filters per layer
layer_filters = [32]


class SAAE():
    def __init__(self, img_shape=input_shape, encoded_dim=latent_dim):
        self.encoded_dim = encoded_dim
        self.optimizer_reconst = Adam(0.001)
        self.optimizer_discriminator = Adam(0.0001)
        self._initAndCompileFullModel(img_shape, encoded_dim)

    def _genEncoderModel(self, img_shape, encoded_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        input_shape = img_shape
        latent_dim = encoded_dim

        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        # stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)

        # shape info needed to build decoder model so we don't do hand computation
        # the input to the decoder's first Conv2DTranspose will have this shape
        # shape is (4, 4, 256) which is processed by the decoder back to (32, 32, 3)
        self.shape = K.int_shape(x)

        # generate a latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector')(x)

        # instantiate encoder model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()
        return encoder

    def _getDecoderModel(self, encoded_dim, img_shape):
        """ Build Decoder Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
            img_shape (tuple) : shape of target images
        Return:
            A sequential keras model
        """
        latent_dim = encoded_dim
        shape = self.shape

        # build the decoder model
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        labels_inputs = Input(shape=(num_labels,))
        concated = concatenate([latent_inputs, labels_inputs])
        x = Dense(shape[1] * shape[2] * shape[3])(concated)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        outputs = Conv2DTranspose(filters=channels,
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        # instantiate decoder model
        decoder = Model([latent_inputs, labels_inputs], outputs, name='decoder')
        decoder.summary()

        return decoder

    def _getDescriminator(self, encoded_dim):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        latent_dim = encoded_dim
        shape = self.shape

        # build the decoder model
        latent_inputs = Input(shape=(latent_dim,), name='discriminator_input')
        x = Dense(shape[1] * shape[2] * shape[3], kernel_initializer=initializer,
                  bias_initializer=initializer)(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = AvgPool2D(pool_size=2, strides=2)(x)
        x = Conv2D(filters=16, kernel_size=3, activation='relu')(x)
        x = AvgPool2D(pool_size=2, strides=2)(x)
        x = Flatten()(x)

        outputs = Dense(1, name='discriminator_output', activation='sigmoid',
                        kernel_initializer=initializer, bias_initializer=initializer)(x)

        # instantiate discriminator model
        discriminator = Model(latent_inputs, outputs, name='discriminator')
        discriminator.summary()

        return discriminator

    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        self.discriminator = self._getDescriminator(encoded_dim)
        img = Input(shape=img_shape)
        label = Input(shape=(num_labels,))
        encoded_repr = self.encoder(img)
        gen_img = self.decoder([encoded_repr, label])
        self.autoencoder = Model([img, label], gen_img)
        valid = self.discriminator(encoded_repr)
        self.encoder_discriminator = Model(img, valid)
        self.discriminator.compile(optimizer=self.optimizer_discriminator,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        self.autoencoder.compile(optimizer=self.optimizer_reconst,
                                 loss='mse')
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.encoder_discriminator.compile(optimizer=self.optimizer_discriminator,
                                           loss='binary_crossentropy',
                                           metrics=['accuracy'])

    def imagegrid(self, epochnumber):
        fig = plt.figure(figsize=[20, 20])
        labels = y_train
        styles = np.random.normal(size=(10, self.encoded_dim))
        k = 0
        for label in labels[epochnumber // num_labels:epochnumber // num_labels + 10]:
            for style in styles:
                img = self.decoder.predict([style.reshape(-1, self.encoded_dim), label.reshape(-1, num_labels)])
                img = img.reshape(input_shape)
                k = k + 1
                # 250 rows, 10 colums
                ax = fig.add_subplot(10, 10, k)
                ax.set_axis_off()
                ax.imshow(img, cmap="gray")
        fig.savefig("images/SAAE/" + str(epochnumber) + ".png")
        plt.show()
        plt.close(fig)

    def train(self, x_train, y_train, batch_size=32, epochs=5000, save_image_interval=100):
        labels = y_train
        for epoch in range(epochs):
            # ---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            labels_batch = labels[idx, :]
            # Generate a half batch of new images
            latent_fake = self.encoder.predict(imgs)
            # gen_imgs = self.decoder.predict(latent_fake)
            latent_real = np.random.normal(size=(batch_size, self.encoded_dim))
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # idx = np.random.randint(0, x_train.shape[0], batch_size)
            # imgs = x_train[idx]
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the autoencode reconstruction
            g_loss_reconstruction = self.autoencoder.train_on_batch([imgs, labels_batch], imgs)

            # Train generator
            g_logg_similarity = self.encoder_discriminator.train_on_batch(imgs, valid_y)
            # Plot the progress
            if (epoch % save_image_interval == 0):
                print("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, d_loss[0], 100 * d_loss[1],
                                                                             g_logg_similarity[1],
                                                                             g_loss_reconstruction))
                self.imagegrid(epoch)


# %%

if __name__ == '__main__':
    ann = SAAE()
    ann.train(x_train, y_train, batch_size, epochs)

# %%



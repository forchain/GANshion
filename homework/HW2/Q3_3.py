
# -*- coding: utf-8 -*-
# Author : Ali Mirzaei
# Date : 19/09/2017


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Conv2DTranspose, concatenate
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam,SGD
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
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten, Activation,Concatenate,LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend, models
#import tensorflow_addons as tfa
import helpers
import tensorflow as tf
print(tf.__version__)

# need to add these for the GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# import the image generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Setting the parameters for training

# batch size and image width to use
batch_size=128
width=100

# all the data directories
train_dir='train/';
test_dir='test/'
valid_dir='valid/';

# the number of epochs
num_epochs=5000

# creating an image generator that will feed the data from
# each of the directories

# we use scaling transformation in this generator
generator=ImageDataGenerator(rescale=1./255)

# we specify the size of the input and batch size
# size of the input is necessary because the image
# needs to be rescaled for the neural network

train_data=generator.flow_from_directory(train_dir, target_size=(width,width),batch_size=10000)
valid_data=generator.flow_from_directory(valid_dir, target_size=(width,width),batch_size=1000)
test_data=generator.flow_from_directory(test_dir, target_size=(width,width),batch_size=1000)

# the number of steps per epoch is samples/batch size
# we need to use these numbers later

train_steps_per_epoch=math.ceil(train_data.samples/batch_size)
valid_steps_per_epoch=math.ceil(valid_data.samples/batch_size)
test_steps_per_epoch=math.ceil(test_data.samples/batch_size)

x_train, y_train = train_data.next()
x_test, y_test = test_data.next()
x_train.shape
num_labels = len(train_data.class_indices)

import matplotlib.pyplot as plt
def show_image(x):
    plt.imshow(np.clip(x, 0, 1))


show_image(x_train[1])


img_rows = width
img_cols = width
channels = 3

# network parameters
input_shape = (img_rows, img_cols, channels)
batch_size = 128
kernel_size = 3
filters = 16
latent_dim = 256

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.ioff()

os.environ['KMP_DUPLICATE_LIB_OK']='True'
class SSAAE():
    def __init__(self, img_shape=input_shape, encoded_dim=latent_dim):
        self.encoded_dim = encoded_dim
        self.optimizer_reconst = Adam(0.0001)
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
        encoder = Sequential()
        encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(encoded_dim))
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
        decoder = Sequential()
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim))
        decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        decoder.summary()
        return decoder

    def _getDescriminator(self, input_dim):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            input_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        latent_input = Input(shape=(input_dim,))
        labels_input = Input(shape=(num_labels + 1,))
        concated = concatenate([latent_input, labels_input])
        discriminator = Sequential()
        discriminator.add(Dense(1000, activation='relu',
                                input_dim=input_dim + num_labels + 1))
        discriminator.add(Dense(1000, activation='relu'))
        discriminator.add(Dense(1, activation='sigmoid'))
        print('debug latent_input:', latent_input.shape)
        print('debug labels_input:', labels_input.shape)
        print('debug concated:', concated.shape)
        print('debug input_dim:', concated.shape)

        out = discriminator(concated)
        discriminator_model = Model([latent_input, labels_input], out)
        discriminator_model.summary()
        return discriminator_model

    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        self.discriminator = self._getDescriminator(encoded_dim)
        img = Input(shape=img_shape)
        label_code = Input(shape=(num_labels + 1,))
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        valid = self.discriminator([encoded_repr, label_code])
        self.encoder_discriminator = Model([img, label_code], valid)
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
        (latent, labels) = self.generateRandomVectors(range(num_labels))
        imgs = self.decoder.predict(latent)
        for index, img in enumerate(imgs):
            img = img.reshape(input_shape)
            ax = fig.add_subplot(num_labels//10, 10, index + 1)
            ax.set_axis_off()
            ax.imshow(img, cmap="gray")
        fig.savefig("images/SSAAE/" + str(epochnumber) + ".png")
        plt.show()
        plt.close(fig)

    def saveLatentMap(self, epochnumber, x_test, y_test):
        fig = plt.figure(figsize=[20, 20])
        lat = self.encoder.predict(x_test)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(lat[:, 0], lat[:, 1], c=y_test)
        fig.savefig("images/SSAAE/map_" + str(epochnumber) + ".png")

    def train(self, x_train, y_train, x_test, y_test, batch_size=32, epochs=10000, save_interval=500):
        for epoch in range(epochs):
            # ---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            y_batch = y_train[idx]
            # Generate a half batch of new images
            latent_fake = self.encoder.predict(imgs)
            # gen_imgs = self.decoder.predict(latent_fake)
            # latent_real = np.random.normal(size=(half_batch, self.encoded_dim))
            (latent_real, labels) = self.generateRandomVectors(y_batch)
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            # Train the discriminator

            d_loss_real = self.discriminator.train_on_batch([latent_real, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([latent_fake, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # idx = np.random.randint(0, x_train.shape[0], batch_size)
            # imgs = x_train[idx]
            # y_batch = y_train[idx]
            # (_, labels) = self.generateRandomVectors(y_batch)
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the autoencode reconstruction
            g_loss_reconstruction = self.autoencoder.train_on_batch(imgs, imgs)

            # Train generator
            g_logg_similarity = self.encoder_discriminator.train_on_batch([imgs, labels], valid_y)
            # Plot the progress
            print("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, d_loss[0], 100 * d_loss[1],
                                                                         g_logg_similarity[1], g_loss_reconstruction))
            if (epoch % save_interval == 0):
                self.imagegrid(epoch)
                self.saveLatentMap(epoch, x_test, y_test)

    def generateRandomVectors(self, y_train):
        vectors = []
        labels = np.zeros((len(y_train), num_labels + 1))

        labels[range(len(y_train)), np.array(y_train).astype(int)] = 1
        for index, y in enumerate(y_train):
            if y == num_labels:
                l = np.random.randint(0, num_labels)
            else:
                l = y
            mean = [num_labels * np.cos((l * 2 * np.pi) / num_labels),
                    num_labels * np.sin((l * 2 * np.pi) / num_labels)]
            v1 = [np.cos((l * 2 * np.pi) / num_labels), np.sin((l * 2 * np.pi) / num_labels)]
            v2 = [-np.sin((l * 2 * np.pi) / num_labels), np.cos((l * 2 * np.pi) / num_labels)]
            a1 = 8
            a2 = .4
            M = np.vstack((v1, v2)).T
            S = np.array([[a1, 0], [0, a2]])
            cov = np.dot(np.dot(M, S), np.linalg.inv(M))
            # cov = cov*cov.T
            vec = np.random.multivariate_normal(mean=mean, cov=cov,
                                                size=latent_dim//2)
            vectors.append(vec)
        return (np.array(vectors).reshape(-1, latent_dim), labels)

if __name__ == '__main__':
    idx_unlabel = np.random.randint(0, x_train.shape[0], 32)

    # convert one hot to integer
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    y_train[idx_unlabel] = num_labels
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    ann = SSAAE()
    #vecs,b = ann.generateRandomVectors(1000*range(10))
    #plt.scatter(vecs[:,0], vecs[:,1])
    ann.train(x_train, y_train, x_test, y_test, batch_size, epochs=num_epochs)
    vecs,b = ann.generateRandomVectors(1000*range(num_labels))
    generated=ann.decoder.predict(vecs)
    print(generated.shape)
    L= helpers.approximateLogLiklihood(generated, x_test)
    print("Log Likelihood")
    print(L)




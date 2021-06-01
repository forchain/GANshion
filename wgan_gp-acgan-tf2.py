# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop, Adam
from functools import partial

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import sys

import tensorflow as tf
import numpy as np
import os
from imageio import imread, imsave, mimsave
import cv2
import glob
from tqdm import tqdm
import pandas as pd
# from tensorflow.keras.losses import binary_crossentropy
# from tensorflow.keras.backend import mean, log, sparse_categorical_crossentropy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
tf.compat.v1.disable_eager_execution()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 100
Z_DIM = 100
WIDTH = 64
HEIGHT = 64
LABEL = 35
SAMPLE_NUM = 999  # 5
OUTPUT_DIR = 'samples'


class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class WGANGP():
    def __init__(self, x_train, y_train
                 , batch_size=BATCH_SIZE
                 , img_rows=WIDTH
                 , img_cols=HEIGHT
                 , channels=3
                 , latent_dim=Z_DIM
                 , num_classes=LABEL
                 ):
        self.x_train = x_train
        self.y_train = y_train

        self.epi = 0.22
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.batch_size = batch_size
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 3
        # optimizer = RMSprop(lr=0.00005)
        optimizer = Adam(learning_rate=0.00002, beta_1=0.5)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.generator.summary()
        self.critic = self.build_critic()
        self.critic.summary()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([z_disc, label])

        # Discriminator determines validity of the real and fake images
        fake, fake_label = self.critic(fake_img)
        valid, valid_label = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated, inter_label = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc, label],
                                  outputs=[valid, fake, validity_interpolated, valid_label, fake_label],
                                  name='critic_model')
        self.critic_model.compile(loss=[
            self.wasserstein_loss,
            self.wasserstein_loss,
            partial_gp_loss,
            'binary_crossentropy',
           'binary_crossentropy'
        ],
            optimizer=optimizer,
            loss_weights=[1, 1, 10, 2, 2])

        self.critic_model.summary()

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        label_zen = Input(shape=(self.num_classes,))
        # Generate images based of noise
        img = self.generator([z_gen, label_zen])
        # Discriminator determines validity
        valid, valid_label = self.critic(img)
        # Defines generator model
        self.generator_model = Model([z_gen, label_zen], [valid, valid_label], name='generator_model')
        self.generator_model.compile(loss=[self.wasserstein_loss, 'binary_crossentropy'],
                                     optimizer=optimizer)
        self.generator_model.summary()

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        from tensorflow.keras.layers import Dense, Reshape, concatenate, Conv2DTranspose

        d = 4
        momentum = 0.9

        model = Sequential(name='generator_body')

        model.add(Dense(d * d * 512, input_dim=self.latent_dim + self.num_classes))
        model.add(Reshape([d, d, 512]))

        model.add(BatchNormalization(momentum=momentum))
        model.add(ReLU())
        model.add(Conv2DTranspose(kernel_size=5, filters=256, strides=2, padding='same'))

        model.add(BatchNormalization(momentum=momentum))
        model.add(ReLU())
        model.add(Conv2DTranspose(kernel_size=5, filters=128, strides=2, padding='same'))

        model.add(BatchNormalization(momentum=momentum))
        model.add(ReLU())
        model.add(Conv2DTranspose(kernel_size=5, filters=64, strides=2, padding='same'))

        model.add(BatchNormalization(momentum=momentum))
        model.add(ReLU())
        model.add(Conv2DTranspose(kernel_size=5, filters=3, strides=2, padding='same', activation='tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))

        model_input = concatenate([noise, label])
        img = model(model_input)

        return Model([noise, label], img, name='generator')

    def build_critic(self):
        model = Sequential(name='critic_body')

        model.add(Conv2D(filters=64, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU())

        model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU())

        model.add(Conv2D(filters=256, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU())

        model.add(Conv2D(filters=512, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU())

        model.add(Flatten())

        model.summary()

        img = Input(shape=self.img_shape)
        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes)(features)

        return Model(img, [validity, label], name='critic')

    def train(self, epochs, sample_interval=50):
        batch_size = self.batch_size

        # Load the dataset

        # Rescale -1 to 1

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                data_index = np.arange(self.x_train.shape[0])
                np.random.shuffle(data_index)
                data_index = data_index[:batch_size]
                imgs = self.x_train[data_index, :, :, :]
                img_labels = self.y_train[data_index, :]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise, img_labels],
                                                          [valid, fake, dummy, img_labels, img_labels])

            # ---------------------
            #  Train Generator
            # ---------------------

            sampled_labels = self.get_random_tags()
            g_loss = self.generator_model.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def get_random_tags(self):
        y = np.random.uniform(0.0, 1.0, [self.batch_size, self.num_classes]).astype(np.float32)
        color = np.random.randint(0, 9)
        length = np.random.randint(9, 14)
        patten = np.random.randint(14, 20)
        neckline = np.random.randint(20, 26)
        sleeve = np.random.randint(26, 30)
        fit = np.random.randint(30, 33)
        occasion = np.random.randint(33, 35)
        for i in range(self.batch_size):

            y[i, :] = 0

            y[i, color] = 1
            y[i, length] = 1
            y[i, patten] = 1
            y[i, neckline] = 1
            y[i, sleeve] = 1
            y[i, fit] = 1
            y[i, occasion] = 1
        return y

    def sample_images(self, epoch):
        all_tags = ['color_beige', 'color_black', 'color_blue', 'color_gray', 'color_green', 'color_pink', 'color_red',
                    'color_white', 'color_yellow',
                    'length_3-4', 'length_knee', 'length_long', 'length_normal', 'length_short',
                    'pattern_floral', 'pattern_lace', 'pattern_polkadots', 'pattern_print', 'pattern_stripes',
                    'pattern_unicolors',
                    'neckline_back', 'neckline_deep', 'neckline_lined', 'neckline_round', 'neckline_v', 'neckline_wide',
                    'sleeve_length_half', 'sleeve_length_long', 'sleeve_length_short', 'sleeve_length_sleeveless',
                    'fit_loose', 'fit_normal', 'fit_tight',
                    'occasion_casual', 'occasion_party']
        tags = ['color_red', 'length_short', 'pattern_unicolors', 'neckline_round', 'sleeve_length_short', 'fit_normal',
                'occasion_casual']
        y_samples = np.zeros([1, self.num_classes])
        for tag in tags:
            y_samples[0, all_tags.index(tag)] = 1
        y_samples = np.repeat(y_samples, self.batch_size, 0)

        z_samples = np.random.uniform(-1.0, 1.0, [self.batch_size, self.latent_dim]).astype(np.float32)

        r, c = 5, 5
        gen_imgs = self.generator.predict([z_samples, y_samples])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/fashion_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    images = glob.glob('dresses/*.jpg')
    if len(images) == 0:
        print('cannot find dresses')
        exit(0)
    images = images[:SAMPLE_NUM]

    tags = pd.read_csv('img_attr_dresses.csv')
    tags.index = tags['img_path']
    tags.head()

    print(len(images))

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    X_all = []
    Y_all = []
    for i in tqdm(range(len(images))):
        image = imread(images[i])
        image = cv2.resize(image, (HEIGHT, WIDTH))
        image = (image / 255. - 0.5) * 2
        X_all.append(image)

        y = list(tags.loc[images[i]])
        Y_all.append(y[1:])

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)
    print(X_all.shape, Y_all.shape)

    wgan = WGANGP(X_all, Y_all)
    wgan.train(epochs=30000, sample_interval=100)

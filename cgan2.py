'''Trains CGAN on MNIST using Keras

CGAN is Conditional Generative Adversarial Network.
This version of CGAN is similar to DCGAN. The difference mainly
is that the z-vector of geneerator is conditioned by a one-hot label
to produce specific fake images. The discriminator is trained to
discriminate real from fake images that are conditioned on
specific one-hot labels.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Mirza, Mehdi, and Simon Osindero. "Conditional generative
adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import glob
import pandas as pd
from imageio import imread, imsave, mimsave
import cv2
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_generator(inputs, labels, image_size):
    model = Sequential(name='generator_body')

    model.add(Dense(256, input_dim=Z_DIM+LABEL))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(image_size * image_size * CHANNELS, activation='tanh'))
    model.add(Reshape((image_size, image_size, CHANNELS)))

    model.summary()

    model_input = concatenate([inputs, labels], axis=1)
    img = model(model_input)

    return Model([inputs, labels], img, name='generator')


def build_discriminator(inputs, labels, image_size):

    model = Sequential(name='discriminator_body')

    model.add(Dense(512, input_dim=image_size * image_size * CHANNELS + LABEL))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    x = inputs
    x = Flatten()(x)

    y = labels
    model_input = concatenate([x, y])

    validity = model(model_input)

    return Model([inputs, labels], validity, name='discriminator')


def get_random_tags(batch_size, num_classes):
    y = np.random.uniform(0.0, 1.0, [batch_size, num_classes]).astype(np.float32)
    for i in range(batch_size):
        color = np.random.randint(0, 9)
        length = np.random.randint(9, 14)
        patten = np.random.randint(14, 20)
        neckline = np.random.randint(20, 26)
        sleeve = np.random.randint(26, 30)
        fit = np.random.randint(30, 33)
        occasion = np.random.randint(33, 35)

        y[i, :] = 0

        y[i, color] = 1
        y[i, length] = 1
        y[i, patten] = 1
        y[i, neckline] = 1
        y[i, sleeve] = 1
        y[i, fit] = 1
        y[i, occasion] = 1
    return y

def train(models, data, params):
    """Train the Discriminator and Adversarial Networks

    Alternately train Discriminator and Adversarial networks by batch.
    Discriminator is trained first with properly labelled real and fake images.
    Adversarial is trained next with fake images pretending to be real.
    Discriminator inputs are conditioned by train labels for real images,
    and random labels for fake images.
    Adversarial inputs are conditioned by random labels.
    Generate sample images per save_interval.

    Arguments:
        models (list): Generator, Discriminator, Adversarial models
        data (list): x_train, y_train data
        params (list): Network parameters

    """
    # the GAN models
    generator, discriminator, adversarial = models
    # images and labels
    x_train, y_train = data
    # network parameters
    batch_size, latent_size, train_steps, num_labels, model_name = params
    # the generator image is saved every 500 steps
    save_interval = 500
    # noise vector to see how the generator output evolves during training
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    # one-hot label the noise will be conditioned to
    noise_class = get_random_tags(16, num_labels)
    # number of elements in train dataset
    train_size = len(x_train)

    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    # create tensorboard graph data for the model
    tb = tf.keras.callbacks.TensorBoard(log_dir='Logs/cgan2',
                                        histogram_freq=0,
                                        batch_size=batch_size,
                                        write_graph=True,
                                        write_grads=False)
    for i in range(train_steps):
        # train the discriminator for 1 batch
        # 1 batch of real (label=1.0) and fake images (label=0.0)
        # randomly pick real images from dataset
        rand_indexes = np.random.randint(0, train_size, size=batch_size)

        real_images = []
        real_labels = []
        for ri in rand_indexes:
            path = x_train[ri]
            image = imread(path)
            image = cv2.resize(image, (HEIGHT, WIDTH))
            image = image.astype('float32') / 255
            real_images.append(image)
            y = list(y_train.loc[path])
            # corresponding one-hot labels of real images
            real_labels.append(y)

        # generate fake images from noise using generator
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # assign random one-hot labels
        fake_labels = get_random_tags(batch_size, num_labels)

        # generate fake images conditioned on fake labels
        fake_images = generator.predict([noise, fake_labels])
        # real + fake images = 1 batch of train data
        x = np.concatenate((real_images, fake_images))
        # real + fake one-hot labels = 1 batch of train one-hot labels
        labels = np.concatenate((real_labels, fake_labels))

        # label real and fake images
        # real images label is 1.0
        y = np.ones([2 * batch_size, 1])
        # fake images label is 0.0
        y[batch_size:, :] = 0.0
        # train discriminator network, log the loss and accuracy
        loss, acc = discriminator.train_on_batch([x, labels], y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        tb.set_model(discriminator)
        tb.on_epoch_end(i, {'d_loss': loss, 'd_acc': acc})

        # train the adversarial network for 1 batch
        # 1 batch of fake images conditioned on fake 1-hot labels 
        # w/ label=1.0
        # since the discriminator weights are frozen in 
        # adversarial network only the generator is trained
        # generate noise using uniform distribution        
        noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
        # assign random one-hot labels
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        # label fake images as real or 1.0
        y = np.ones([batch_size, 1])
        # train the adversarial network 
        # note that unlike in discriminator training, 
        # we do not save the fake images in a variable
        # the fake images go to the discriminator input
        # of the adversarial for classification
        # log the loss and accuracy
        loss, acc = adversarial.train_on_batch([noise, fake_labels], y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        tb.set_model(adversarial)
        tb.on_epoch_end(i, {'g_loss': loss, 'g_acc': acc})
        print(log)
        if (i + 1) % save_interval == 0:
            # plot generator images on a periodic basis
            plot_images(generator,
                        noise_input=noise_input,
                        noise_class=noise_class,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)

    tb.on_train_end(None)

    # save the model after training the generator
    # the trained generator can be reloaded for 
    # future MNIST digit generation
    generator.save(model_name + ".h5")


def plot_images(generator,
                noise_input,
                noise_class,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    Arguments:
        generator (Model): The Generator Model for fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name

    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict([noise_input, noise_class])
    print(model_name , " labels for generated images: ", np.argmax(noise_class, axis=1))
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size, CHANNELS])
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


BATCH_SIZE = 64
Z_DIM = 100
WIDTH = 64
HEIGHT = 64
LABEL = 35
SAMPLE_NUM = 999  # 5
OUTPUT_DIR = 'samples'
EPOCHS = 10000
CHANNELS = 3


def build_and_train_models():
    images = glob.glob('dresses/*.jpg')
    if len(images) == 0:
        print('cannot find dresses')
        exit(0)

    tags = pd.read_csv('img_attr_dresses.csv')
    tags.index = tags['img_path']
    tags = tags.drop(columns=['img_path'])
    tags.head()

    print(len(images))

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    x_train, y_train = images, tags

    # reshape data for CNN as (28, 28, 1) and normalize
    # image_size = x_train.shape[1]
    #
    # x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    # x_train = x_train.astype('float32') / 255

    num_labels = LABEL
    image_size = WIDTH
    # y_train = to_categorical(y_train)

    model_name = "cgan_2"
    # network parameters
    # the latent or z vector is 100-dim
    latent_size = Z_DIM
    batch_size = BATCH_SIZE
    train_steps = EPOCHS
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, CHANNELS)
    label_shape = (num_labels, )

    # build discriminator model
    inputs = Input(shape=input_shape, name='discriminator_input')
    labels = Input(shape=label_shape, name='class_labels')

    # discriminator = build_discriminator(inputs, labels, image_size)
    discriminator = build_discriminator(inputs, labels, image_size)
    # [1] or original paper uses Adam,
    # but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # build generator model
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    # generator = build_generator(inputs, labels, image_size)
    generator = build_generator(inputs, labels, image_size)
    generator.summary()

    # build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    # freeze the weights of discriminator during adversarial training
    discriminator.trainable = False
    outputs = discriminator([generator([inputs, labels]), labels])
    adversarial = Model([inputs, labels],
                        outputs,
                        name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    plot_images(generator,
                noise_input=noise_input,
                noise_class=noise_class,
                show=True,
                step=step,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        class_label = None
        if args.digit is not None:
            class_label = args.digit
        test_generator(generator, class_label)
    else:
        build_and_train_models()

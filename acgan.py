from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import ReLU, LeakyReLU, concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import glob
import os
from imageio import imread
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras import metrics

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 64
Z_DIM = 100
WIDTH = 64
HEIGHT = 64
LABEL = 35
SAMPLE_NUM = 999  # 5
OUTPUT_DIR = 'acgan'
EPOCHS = 14000
CHANNELS = 3


def F1_Score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


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


class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = WIDTH
        self.img_cols = HEIGHT
        self.channels = CHANNELS
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = LABEL
        self.latent_dim = Z_DIM

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'binary_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator_3()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy',
                                            metrics.AUC(name='auc'),
                                            metrics.Precision(name='precision'),
                                            metrics.Recall(name='recall'),
                                            F1_Score
                                            ])
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator_3()
        self.generator.summary()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer,
                              metrics=['accuracy',
                                       metrics.AUC(name='auc'),
                                       metrics.Precision(name='precision'),
                                       metrics.Recall(name='recall'),
                                       F1_Score
                                       ])

    def build_generator_1(self):

        model = Sequential(name='generator_body')

        size = self.img_rows // 4
        model.add(Dense(128 * size * size, activation="relu", input_dim=self.latent_dim + self.num_classes))
        model.add(Reshape((size, size, 128)))

        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=128, strides=2, kernel_size=5, padding="same", activation='relu'))

        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=64, strides=2, kernel_size=5, padding="same", activation='relu'))

        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=32, strides=1, kernel_size=5, padding="same", activation='relu'))

        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=3, strides=1, kernel_size=5, padding="same", activation='relu'))

        model.add(Activation("sigmoid"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))

        model_input = concatenate([noise, label], axis=1)
        img = model(model_input)

        return Model([noise, label], img, name='generator')

    def build_generator_2(self):

        model = Sequential(name='generator_body')

        size = self.img_rows // 4
        model.add(Dense(512 * size * size, activation="relu", input_dim=self.latent_dim + self.num_classes))
        model.add(Reshape((size, size, 512)))

        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())
        model.add(Conv2DTranspose(filters=256, strides=2, kernel_size=5, padding="same"))

        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())
        model.add(Conv2DTranspose(filters=128, strides=2, kernel_size=5, padding="same"))

        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())
        model.add(Conv2DTranspose(filters=64, strides=1, kernel_size=5, padding="same"))

        model.add(BatchNormalization(momentum=0.9))
        model.add(ReLU())
        model.add(Conv2DTranspose(filters=3, strides=1, kernel_size=5, padding="same"))

        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))

        model_input = concatenate([noise, label], axis=1)
        img = model(model_input)

        return Model([noise, label], img, name='generator')

    def build_generator_3(self):
        model = Sequential(name='generator_body')

        image_size = self.img_rows
        model.add(Dense(256, input_dim=Z_DIM + LABEL))
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

        inputs = Input(shape=(self.latent_dim,))
        labels = Input(shape=(self.num_classes,))

        model_input = concatenate([inputs, labels], axis=1)
        img = model(model_input)

        return Model([inputs, labels], img, name='generator')

    def build_discriminator_1(self):

        model = Sequential(name='discriminator_body')

        model.add(LeakyReLU(alpha=0.2, input_shape=self.img_shape))
        model.add(Conv2D(filters=32, kernel_size=5, strides=2, padding="same"))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding="same"))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding="same"))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding="same"))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1)(features)
        validity = Activation('sigmoid')(validity)
        # label = Dense(self.num_classes, activation="softmax")(features)
        layer = Dense(128)(features)

        label = Dense(self.num_classes)(layer)

        label = Activation("sigmoid", name='label')(label)

        return Model(img, [validity, label], name='discriminator')

    def build_discriminator_2(self):

        model = Sequential(name='discriminator_body')

        model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding="same", input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=256, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=512, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1)(features)
        # label = Dense(self.num_classes, activation="softmax")(features)

        label = Dense(self.num_classes)(features)

        return Model(img, [validity, label], name='discriminator')

    def build_discriminator_3(self):

        model = Sequential(name='discriminator_body')
        image_size = self.img_rows
        model.add(Dense(512, input_dim=image_size * image_size * CHANNELS))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        # model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        x = Flatten()(img)
        features = model(x)

        # Determine validity and label of the image
        validity = Dense(1, activation='sigmoid')(features)

        label = Dense(self.num_classes, activation='sigmoid')(features)

        return Model(img, [validity, label], name='discriminator')

    def build_generator(self):

        model = Sequential(name='generator_body')

        size = self.img_rows // 4
        model.add(Dense(128 * size * size, activation="relu", input_dim=self.latent_dim + self.num_classes))
        model.add(Reshape((size, size, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))

        model_input = concatenate([noise, label], axis=1)
        img = model(model_input)

        return Model([noise, label], img, name='generator')

    def build_discriminator(self):

        model = Sequential(name='discriminator_body')

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        # label = Dense(self.num_classes, activation="softmax")(features)
        label = Dense(self.num_classes, activation="sigmoid")(features)

        return Model(img, [validity, label], name='discriminator')

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
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

        X_train, y_train = images, tags

        # # Configure inputs
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        # y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # create tensorboard graph data for the model
        tb = tf.keras.callbacks.TensorBoard(log_dir='Logs/acgan',
                                            histogram_freq=0,
                                            batch_size=batch_size,
                                            write_graph=True,
                                            write_grads=False)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, len(X_train), batch_size)
            imgs = []
            img_labels = []
            for ri in idx:
                path = X_train[ri]
                image = imread(path)
                image = cv2.resize(image, (HEIGHT, WIDTH))
                image = (image.astype('float32') - 127.5) / 127.5
                imgs.append(image)
                y = list(y_train.loc[path])
                # corresponding one-hot labels of real images
                img_labels.append(y)

            imgs = np.asarray(imgs)
            img_labels = np.array(img_labels)

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = get_random_tags(batch_size, self.num_classes)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            tb.set_model(self.discriminator)
            tb.on_epoch_end(epoch, {'d_loss': d_loss[0], 'd_acc': d_loss[3]})

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
            tb.set_model(self.combined)
            tb.on_epoch_end(epoch, {'g_loss': g_loss[0], 'g_acc': g_loss[3]})

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (
                epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0], 100 * g_loss[3]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = get_random_tags(r * c, LABEL)
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(OUTPUT_DIR + "/%d.png" % epoch)
        plt.close()

    def save_model(self):

        model_dir = OUTPUT_DIR + '/saved_model'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        def save(model, model_name):
            model_path = model_dir + "/%s.json" % model_name
            weights_path = model_dir + "/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=200)

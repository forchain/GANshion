# %%

# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from imageio import imread, imsave, mimsave
import cv2
import glob
from tqdm import tqdm
import pandas as pd

from tensorflow.keras.layers import LeakyReLU, Conv2D, Flatten, Dense, Input
from tensorflow.keras.layers import BatchNormalization, Reshape, Conv2DTranspose, concatenate, ReLU
from tensorflow.keras.models import Sequential, Model

tf.compat.v1.disable_eager_execution()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%

images = glob.glob('dresses/*.jpg')
if len(physical_devices) == 0:
    images = images[:1000]

print(len(images))

# %%
# 加载标签
tags = pd.read_csv('img_attr_dresses.csv')
tags.index = tags['img_path']
tags.head()

# %%

BATCH_SIZE = 100
Z_DIM = 100
WIDTH = 64
HEIGHT = 64
LABEL = 35
LAMBDA = 10
DIS_ITERS = 3  # 5

OUTPUT_DIR = 'wgan_gp-acgan'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

img_shape = (HEIGHT, WIDTH, 3)

X = Input(shape=img_shape, name='X')
Y = Input(shape=(LABEL,), name='Y')
noise = Input(shape=(Z_DIM,), name='noise')


def build_discriminator():
    model = Sequential(name='discriminator_body')

    model.add(Conv2D(filters=64, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(filters=256, kernel_size=5, strides=2, padding="same"))
    model.add(LeakyReLU())

    model.add(Conv2D(filters=512, kernel_size=5, strides=2, padding="same"))
    model.add(LeakyReLU())

    model.add(Flatten())

    model.summary()

    img = Input(shape=img_shape)
    features = model(img)

    validity = Dense(1)(features)
    label = Dense(LABEL)(features)

    return Model(img, [validity, label], name='discriminator')


# %%

def build_generator():
    d = 4
    momentum = 0.9

    model = Sequential(name='generator_body')

    model.add(Dense(d * d * 512, input_dim=Z_DIM + LABEL))
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
    model.add(Conv2DTranspose(kernel_size=5, filters=3, strides=2, padding='same', activation='tanh', name='g'))

    model.summary()

    noise = Input(shape=(Z_DIM,))
    label = Input(shape=(LABEL,))

    model_input = concatenate([noise, label])
    img = model(model_input)

    return Model([noise, label], img, name='generator')


# %%


generator = build_generator()
g = generator(noise, Y)
discriminator = build_discriminator()
d_real, y_real = discriminator(X)
d_fake, y_fake = discriminator(g)


loss_d_real = -tf.reduce_mean(input_tensor=d_real)
loss_d_fake = tf.reduce_mean(input_tensor=d_fake)

loss_cls_real = tf.compat.v1.losses.mean_squared_error(Y, y_real)
loss_cls_fake = tf.compat.v1.losses.mean_squared_error(Y, y_fake)

loss_d = loss_d_real + loss_d_fake + loss_cls_real
loss_g = -tf.reduce_mean(input_tensor=d_fake) + loss_cls_fake

alpha = tf.random.uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
interpolates = alpha * X + (1 - alpha) * g
grad = tf.gradients(ys=discriminator(interpolates, reuse=True), xs=[interpolates])[0]
slop = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(grad), axis=[1]))
gp = tf.reduce_mean(input_tensor=(slop - 1.) ** 2)
loss_d += LAMBDA * gp

vars_g = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('discriminator')]

# %%

update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)

generator.compile

# %%

def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


# %%

# 整理数据
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

# %%

for i in range(10):
    plt.imshow((X_all[i, :, :, :] + 1) / 2)
    plt.show()
print(Y_all[i, :])


# %%

def get_random_batch():
    data_index = np.arange(X_all.shape[0])
    np.random.shuffle(data_index)
    data_index = data_index[:BATCH_SIZE]
    X_batch = X_all[data_index, :, :, :]
    Y_batch = Y_all[data_index, :]

    return X_batch, Y_batch


# %%

if __name__ == '__main__':
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    zs = np.random.uniform(-1.0, 1.0, [BATCH_SIZE // 2, Z_DIM]).astype(np.float32)

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
    y_samples = np.zeros([1, LABEL])
    for tag in tags:
        y_samples[0, all_tags.index(tag)] = 1
    y_samples = np.repeat(y_samples, BATCH_SIZE, 0)

    z_samples = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, Z_DIM]).astype(np.float32)
    samples = []
    loss = {'d': [], 'g': []}

    for i in tqdm(range(10000)):
        for j in range(DIS_ITERS):
            n = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, Z_DIM]).astype(np.float32)
            X_batch, Y_batch = get_random_batch()
            _, d_ls = sess.run([optimizer_d, loss_d], feed_dict={X: X_batch, Y: Y_batch, noise: n, is_training: True})

        _, g_ls = sess.run([optimizer_g, loss_g], feed_dict={X: X_batch, Y: Y_batch, noise: n, is_training: True})

        loss['d'].append(d_ls)
        loss['g'].append(g_ls)

        if i % 500 == 0:
            print(i, d_ls, g_ls)
            gen_imgs = sess.run(g, feed_dict={noise: z_samples, Y: y_samples, is_training: False})
            gen_imgs = (gen_imgs + 1) / 2
            imgs = [img[:, :, :] for img in gen_imgs]
            gen_imgs = montage(imgs)
            plt.axis('off')
            plt.imshow(gen_imgs)
            imsave(os.path.join(OUTPUT_DIR, 'fashion_%d.jpg' % i), gen_imgs)
            plt.show()
            samples.append(gen_imgs)

    plt.plot(loss['d'], label='Discriminator')
    plt.plot(loss['g'], label='Generator')
    plt.legend(loc='upper right')
    plt.savefig(OUTPUT_DIR + '/Loss.png')
    plt.show()
    mimsave(os.path.join(OUTPUT_DIR, 'fashion.gif'), samples, fps=10)

    # %%

    saver = tf.compat.v1.train.Saver()
    saver.save(sess, OUTPUT_DIR + '/model', global_step=60000)

    # %%

# %%

# 加载库
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from imageio import imread, imsave, mimsave
import cv2
import glob
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%

# 加载图片
images = glob.glob('dresses/*.jpg')
print(len(images))

# %%

# 加载标签
tags = pd.read_csv('img_attr_dresses.csv')
tags.index = tags['img_path']
tags.head()

# %%

# 定义一些常量、网络tensor、辅助函数，批大小设为2的幂比较合适，这里设为64，考虑学习率衰减
batch_size = 64
z_dim = 128
WIDTH = 128
HEIGHT = 128
LABEL = 35
LAMBDA = 0.05
BETA = 3

OUTPUT_DIR = 'samples'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, HEIGHT, WIDTH, 3], name='X')
X_perturb = tf.placeholder(dtype=tf.float32, shape=[batch_size, HEIGHT, WIDTH, 3], name='X_perturb')
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, LABEL], name='Y')
noise = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='noise')
noise_y = tf.placeholder(dtype=tf.float32, shape=[batch_size, LABEL], name='noise_y')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

global_step = tf.Variable(0, trainable=False)
add_global = global_step.assign_add(1)
initial_learning_rate = 0.0002
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=20000,
                                           decay_rate=0.5)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


def conv2d(inputs, kernel_size, filters, strides, padding='same', use_bias=True):
    return tf.layers.conv2d(inputs=inputs, kernel_size=kernel_size, filters=filters, strides=strides, padding=padding,
                            use_bias=use_bias)


def batch_norm(inputs, is_training=is_training, decay=0.9):
    return tf.contrib.layers.batch_norm(inputs, is_training=is_training, decay=decay)


# %%

# 判别器部分
def d_block(inputs, filters):
    h0 = lrelu(conv2d(inputs, 3, filters, 1))
    h0 = conv2d(h0, 3, filters, 1)
    h0 = lrelu(tf.add(h0, inputs))
    return h0


def discriminator(image, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = image

        f = 32
        for i in range(5):
            if i < 3:
                h0 = lrelu(conv2d(h0, 4, f, 2))
            else:
                h0 = lrelu(conv2d(h0, 3, f, 2))
            h0 = d_block(h0, f)
            h0 = d_block(h0, f)
            f = f * 2

        h0 = lrelu(conv2d(h0, 3, f, 2))
        h0 = tf.contrib.layers.flatten(h0)
        Y_ = tf.layers.dense(h0, units=LABEL)
        h0 = tf.layers.dense(h0, units=1)
        return h0, Y_


# %%

# 生成器部分
def g_block(inputs):
    h0 = tf.nn.relu(batch_norm(conv2d(inputs, 3, 64, 1, use_bias=False)))
    h0 = batch_norm(conv2d(h0, 3, 64, 1, use_bias=False))
    h0 = tf.add(h0, inputs)
    return h0


def generator(z, label):
    with tf.variable_scope('generator', reuse=None):
        d = 16
        z = tf.concat([z, label], axis=1)
        h0 = tf.layers.dense(z, units=d * d * 64)
        h0 = tf.reshape(h0, shape=[-1, d, d, 64])
        h0 = tf.nn.relu(batch_norm(h0))
        shortcut = h0

        for i in range(16):
            h0 = g_block(h0)

        h0 = tf.nn.relu(batch_norm(h0))
        h0 = tf.add(h0, shortcut)

        for i in range(3):
            h0 = conv2d(h0, 3, 256, 1, use_bias=False)
            h0 = tf.depth_to_space(h0, 2)
            h0 = tf.nn.relu(batch_norm(h0))

        h0 = tf.layers.conv2d(h0, kernel_size=9, filters=3, strides=1, padding='same', activation=tf.nn.tanh, name='g',
                              use_bias=True)
        return h0


# %%

# 损失函数，这里的gp项来自DRAGAN，https://arxiv.org/abs/1705.07215，WGAN使用真实样本和合成样本的插值，
# 而DRAGAN使用真实样本和干扰样本的插值
g = generator(noise, noise_y)
d_real, y_real = discriminator(X)
d_fake, y_fake = discriminator(g, reuse=True)

loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real, tf.ones_like(d_real)))
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake, tf.zeros_like(d_fake)))
loss_g_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake, tf.ones_like(d_fake)))

loss_c_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(y_real, Y))
loss_c_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(y_fake, noise_y))

loss_d = loss_d_real + loss_d_fake + BETA * loss_c_real
loss_g = loss_g_fake + BETA * loss_c_fake

alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
interpolates = alpha * X + (1 - alpha) * X_perturb
grad = tf.gradients(discriminator(interpolates, reuse=True)[0], [interpolates])[0]
slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
gp = tf.reduce_mean((slop - 1.) ** 2)
loss_d += LAMBDA * gp

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

# %%

# 定义优化器
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(loss_g, var_list=vars_g)


# %%

# 合成图片的函数
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

# 定义随机产生标签的函数，原始数据中标签分布不均匀，但我们希望 [公式] 能学到各种标签，所以均匀地生成各类标签
def get_random_tags():
    y = np.random.uniform(0.0, 1.0, [batch_size, LABEL]).astype(np.float32)
    for i in range(batch_size):
        color = np.random.randint(0, 9)
        length = np.random.randint(9, 14)
        patten = np.random.randint(14, 20)
        neckline = np.random.randint(20, 26)
        sleeve = np.random.randint(26, 30)
        fit = np.random.randint(30, 33)
        occasion = np.random.randint(33, 35)

        y[i, :] = 0

        y[color] = 1
        y[length] = 1
        y[patten] = 1
        y[neckline] = 1
        y[sleeve] = 1
        y[fit] = 1
        y[occasion] = 1
    return y


# %%

# 训练模型，CelebA中男女比例均衡，因此每次迭代随机取一批数据训练即可。
# 但现在由于原始数据中各类标签分布不均匀，所以需要完整地迭代数据

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    y_samples = get_random_tags()
    samples = []
    loss = {'d': [], 'g': []}

    offset = 0
    for i in tqdm(range(60000)):
        if offset + batch_size > X_all.shape[0]:
            offset = 0
        if offset == 0:
            data_index = np.arange(X_all.shape[0])
            np.random.shuffle(data_index)
            X_all = X_all[data_index, :, :, :]
            Y_all = Y_all[data_index, :]
        X_batch = X_all[offset: offset + batch_size, :, :, :]
        Y_batch = Y_all[offset: offset + batch_size, :]
        X_batch_perturb = X_batch + 0.5 * X_batch.std() * np.random.random(X_batch.shape)
        offset += batch_size

        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        ny = get_random_tags()
        _, d_ls = sess.run([optimizer_d, loss_d],
                           feed_dict={X: X_batch, X_perturb: X_batch_perturb, Y: Y_batch, noise: n, noise_y: ny,
                                      is_training: True})

        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        ny = get_random_tags()
        _, g_ls = sess.run([optimizer_g, loss_g], feed_dict={noise: n, noise_y: ny, is_training: True})

        loss['d'].append(d_ls)
        loss['g'].append(g_ls)

        _, lr = sess.run([add_global, learning_rate])

        if i % 500 == 0:
            print(i, d_ls, g_ls, lr)
            gen_imgs = sess.run(g, feed_dict={noise: z_samples, noise_y: y_samples, is_training: False})
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
    plt.savefig('Loss.png')
    plt.show()
    mimsave(os.path.join(OUTPUT_DIR, 'fashion.gif'), samples, fps=10)

    # %%

    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, './fashion_acgan', global_step=60000)

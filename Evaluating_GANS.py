# %%

# Evaluating GANS
## https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from numpy import asarray


# %%

# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# %%

from tensorflow.keras.datasets import cifar10

# load the CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('uint8')

# need to reshape to make sure that subset works for multidimensional
# array

idx5 = (y_train == 5).reshape(x_train.shape[0])
idx6 = (y_train == 6).reshape(x_train.shape[0])
idx7 = (y_train == 7).reshape(x_train.shape[0])

x_train_5 = x_train[idx5]
x_train_6 = x_train[idx6]
x_train_7 = x_train[idx7]

# for i in range(1, 3):
#     plt.imshow(x_train_5[i])
#     plt.show()
#     plt.imshow(x_train_6[i])
#     plt.show()
#     plt.imshow(x_train_7[i])
#     plt.show()

# %%

# convert each image into linear from 32x32x3
x_train_5_flat = x_train_5.reshape((-1, 32 * 32 * 3))
x_train_6_flat = x_train_6.reshape((-1, 32 * 32 * 3))
x_train_7_flat = x_train_7.reshape((-1, 32 * 32 * 3))

print(x_train_5_flat.shape)

# %%

# randomly select 10 elements from each

indices = np.random.choice(x_train_5.shape[0], 10, replace=False)
print(indices)

x_train_5_10 = x_train_5_flat[indices]
x_train_6_10 = x_train_6_flat[indices]
x_train_7_10 = x_train_7_flat[indices]

print(x_train_5_10.shape)

# %%

from tqdm import tqdm

fid_5_7 = calculate_fid(x_train_5_10, x_train_7_10)
# ValueError: m has more than 2 dimensions
# fid_5_5 = calculate_fid(x_train_5, x_train_5)
fid_5_5 = calculate_fid(x_train_5_10, x_train_5_10)

print('fid between 5 and 7', fid_5_7)
print('fid between 5 and 5', fid_5_5)

# %%

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.datasets import cifar10

# %%

#!conda install scikit-image

# %%

from skimage.transform import resize


# need to first make it the right size
# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# %%

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# %%

# create a small subset of the image sets
indices = np.random.choice(x_train_5.shape[0], 25, replace=False)

x_train_5_small = x_train_5[indices]
x_train_7_small = x_train_7[indices]

# %%

# proprocess the input data -- first we need
# to convert to 299,299,3

print('Initial', x_train_5_small.shape, x_train_7_small.shape)
# convert integer to floating point values
# because we need to interpolate
x_train_5_small = x_train_5_small.astype('float32')
x_train_7_small = x_train_7_small.astype('float32')
# resize images
x_train_5_small = scale_images(x_train_5_small, (299, 299, 3))
x_train_7_small = scale_images(x_train_7_small, (299, 299, 3))
print('Scaled', x_train_5_small.shape, x_train_7_small.shape)

for i in range(1, 3):
    plt.imshow(x_train_5_small[i].astype(int))
    plt.show()
    plt.imshow(x_train_7_small[i].astype(int))
    plt.show()

# %%

# now we will pre-process for the inception3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# pre-process images
x_train_5_small = preprocess_input(x_train_5_small)
x_train_7_small = preprocess_input(x_train_7_small)

# %%

x_train_5_prediction = model.predict(x_train_5_small)
print(x_train_5_prediction.shape)


# %%

# calculate frechet inception distance
def calculate_fid(model, img1, img2):
    act1 = model.predict(img1)
    act2 = model.predict(img2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# %%

# calculate fid
fid = calculate_fid(model, x_train_5_small, x_train_7_small)
print('FID: %.3f' % fid)

# %%

# load the full model with the head in intact
# prepare the inception v3 model
full_model = InceptionV3(include_top=True, pooling='avg', input_shape=(299, 299, 3))

# %%

predicted_values = full_model.predict(x_train_5_small)
print(predicted_values[0])

# %%

# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp


# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score


# %%

# conditional probabilities for high quality images
p_yx_a = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
p_yx_b = asarray([[0.5, 0.0, 0.5], [0.33, 0.33, 0.33], [0.0, 0.0, 1.0]])

iscore_a = calculate_inception_score(p_yx_a)
iscore_b = calculate_inception_score(p_yx_b)

print('IS for a=%.3f and for b= %.3f ' % (iscore_a, iscore_b))

# %%

# nsplits

n_split = 5
n_part = np.floor(x_train_5_small.shape[0] / n_split)
print(n_part)

# %%

import math
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


# assumes images have the shape 299x299x3, pixels in [0,255]
# assume the data has been scaled and pre-processed
def calculate_inception_score(images, n_split=5, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # predict class probabilities for images
    yhat = model.predict(images)
    # enumerate splits of images/predictions
    scores = list()
    # calculate the score in batches
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve p(y|x) these are [0, 0.5, 0, 0.5], etc.
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = yhat[ix_start:ix_end]
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std


# %%

# calculate the inception score

iscore = calculate_inception_score(x_train_5_small)
print(iscore)

# %%

# using KNN = 1

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# we pretend that the dogs are real
# and horses are fake generated by GANs

n = 500

x_real = x_train_5_flat[0:n]
x_fake = x_train_7_flat[0:n]

y_real = np.ones(x_real.shape[0])
y_fake = np.zeros(x_fake.shape[0])

x = np.vstack((x_real, x_fake))
y = np.hstack((y_real, y_fake))

print(x.shape)
print(y.shape)

# %%

from sklearn.neighbors import KNeighborsClassifier

# %%

knn = KNeighborsClassifier(n_neighbors=1)

# %%

knn_model = knn.fit(x, y)

# %%

from sklearn.metrics import accuracy_score

predicted = knn_model.predict(x)
print(accuracy_score(predicted, y))

# %%

print(knn.score(x, y))

# %%

# using euclidian distance
from sklearn.model_selection import LeaveOneOut

# %%

from tqdm import tqdm

loo = LeaveOneOut()

scores = list()
for train, test in tqdm(loo.split(x)):
    # fit the model leaving one out
    knn_model = knn.fit(x[train], y[train])
    predicted = knn_model.predict(x[test])
    actual = y[test]
    scores.append(accuracy_score(actual, predicted))

# %%

print(np.array(scores).mean())
print(np.array(scores).std())


# %%

# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# %%

nbrs = KNeighborsClassifier(n_neighbors=1, metric='pyfunc', metric_params={"func": calculate_psnr})
knn_model = nbrs.fit(x, y)

# %%

from tqdm import tqdm

loo = LeaveOneOut()

scores = list()
for train, test in tqdm(loo.split(x)):
    # fit the model leaving one out
    knn_model = nbrs.fit(x[train], y[train])
    predicted = knn_model.predict(x[test])
    actual = y[test]
    scores.append(accuracy_score(actual, predicted))

# %%

print(np.array(scores).mean())
print(np.array(scores).std())

# %%



import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd

from scipy.linalg import sqrtm
from scipy import ndimage as ndi
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from skimage.transform import resize
from skimage.util import img_as_ubyte

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.models import Model

autotune = tf.data.experimental.AUTOTUNE


class ReflectionPadding2D(layers.Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


# 8 a callback to periodically saves generated images
class GANMonitor(keras.callbacks.Callback):


    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        for i, img in enumerate(test_base.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save("generated_img{epoch}_{i}.png".format(epoch=epoch + 1, i=i))



adv_loss_fn = keras.losses.MeanSquaredError()

# 11  Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# 12 Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


def segment_lung_mask(img, value):
    # step 0: remove dimension
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # step 1: convert into binary image
    binary = np.array(img < value, dtype=np.int8)

    # step 2: clear border
    cleared = clear_border(binary)

    # step 3: label the image
    label_image = label(cleared)

    # step 4: keep the labels with 2 largest areas
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    # step 5: erosion operation with a disk of radius 2 to separate the lung nodules attached to the blood vessels
    selem = disk(2)
    binary = binary_erosion(binary, selem)

    # step 6: clossure operation with a disk of radius 10, to keep nodules attached to the lung wall
    selem = disk(10)
    binary = binary_closing(binary, selem)

    # step 7: fill in the small holes inside the binary mask of the lungs
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    inverse_mask = 1 - binary
    img_inverse = img.copy()

    # step 8: superimposed
    get_high_vals = binary == 0
    img[get_high_vals] = 0

    get_values = inverse_mask == 0
    img_inverse[get_values] = 0

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2RGB)
    return img, img_inverse


def make_prediction(n, samples):
    predictions = []
    for i, img in enumerate(samples.take(n)):
        prediction = cycle_gan_model.gen_G(img, training=True)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        prediction = keras.preprocessing.image.array_to_img(prediction)
        predictions.append(prediction)

    return predictions


def make_prediction_save_images(model, X, num, path, patient):
    predictions = []
    for i, img in enumerate(X.take(num)):
        prediction = model(img, training=True)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

        predictions.append(prediction)
        img = keras.preprocessing.image.array_to_img(prediction)
        img.save(str(path) + "slice_{j}_img_{i}.jpg".format(i=i, j=patient))

    return predictions

def calculate_fid(model, img_1, img_2):
    fid_arr = []
    # calculate activations
    act_1 = model.predict(img_1)
    act_2 = model.predict(img_2)

    # calculate mean and covariance statistic
    mu_1, sigma_1 = act_1.mean(axis=0), np.cov(act_1, rowvar=False)
    mu_2, sigma_2 = act_2.mean(axis=0), np.cov(act_2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu_1 - mu_2) ** 2)

    # calculate sum squared difference between means
    covmean = sqrtm(sigma_1.dot(sigma_2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma_1 + sigma_2 - 2.0 * covmean)

    return fid



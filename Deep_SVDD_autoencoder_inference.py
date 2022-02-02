import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from natsort import natsorted

import tensorflow_addons as tfa

import cv2
import shutil
import glob

import tensorflow as tf
from tensorflow.keras import layers, models, Input

import sklearn

image_shape = (28,28,1)

img_dim = image_shape[0]

class CAE(tf.keras.Model):
    """Convolutional autoencoder."""
    def __init__(self, latent_dim):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim
        self.LeakyReLU_rate = 0.01
        self.BatchNormalization_trainable_tag = False
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=5, strides=(1, 1), padding='same', use_bias=False),
                # tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=self.BatchNormalization_trainable_tag),
                tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.MaxPool2D(),

                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=5, strides=(1, 1), padding='same', use_bias=False),
                # tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=self.BatchNormalization_trainable_tag),
                tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.MaxPool2D(),

                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim, use_bias=False),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*4, use_bias=False),
                # tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=self.BatchNormalization_trainable_tag),
                tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.Reshape(target_shape=(7, 7, 4)),

                tf.keras.layers.Conv2DTranspose(
                    filters=4, kernel_size=5, strides=1, padding='same', use_bias=False),
                # tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=self.BatchNormalization_trainable_tag),
                tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.UpSampling2D(),

                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=5, strides=1, padding='same', use_bias=False),
                # tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=self.BatchNormalization_trainable_tag),
                tf.keras.layers.LeakyReLU(alpha=self.LeakyReLU_rate),
                tf.keras.layers.UpSampling2D(),

                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=5, strides=1, padding='same', use_bias=False),
            ]
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    # def decode(self, z, apply_sigmoid=False):
    def decode(self, z, apply_sigmoid=True):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

latent_dim = 32
model = CAE(latent_dim)

def dataset_collection_func(normal_class):

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    one_class_idx = np.where(train_labels == normal_class)
    train_images = train_images[one_class_idx]
    
    return train_images, test_images

train_images, test_images = dataset_collection_func(normal_class = 8)

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = './Deep_SVDD_checkpoints'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

checkpoint.restore(ckpt_manager.latest_checkpoint) # restore the lastest checkpoints

def generate_and_save_images(model, test_sample):
    z = model.encode(test_sample)
    predictions = model.decode(z)

    # print(predictions.shape)

    fig = plt.figure(figsize=(1, 2))

    # print(predictions.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(test_sample[0, :, :, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(predictions[0, :, :, 0])

    plt.show()

def inference(data_paths, i):
    
    img_array = data_paths[i]

    img_array = np.array(img_array)
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    image_batch = img_array

    generate_and_save_images(model, test_sample=image_batch)

inference(train_images, 50)
# inference(test_images, 55)
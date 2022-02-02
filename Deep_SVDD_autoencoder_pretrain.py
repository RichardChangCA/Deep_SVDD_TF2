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

def compute_loss(x):
    z = model.encode(x)
    reconstruction = model.decode(z)

    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            (x - reconstruction) ** 2, axis=(1, 2)
        )
    )

    return reconstruction_loss

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

def train_step(inputs_image, optimizer):
    # print("training......")

    with tf.GradientTape() as tape:
        loss = compute_loss(inputs_image)

        # print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.array(loss), optimizer

def train(train_images, test_images, epochs, BATCH_SIZE):

    global learning_rate
    global optimizer

    for epoch in range(epochs):
        start = time.time()

        idx = np.random.permutation(len(train_images))
        train_images = train_images[idx]

        for index in range(0, len(train_images)-BATCH_SIZE, BATCH_SIZE):
            label_batch = []

            for i in range(BATCH_SIZE):

                img_array = train_images[index+i]
                img_array = np.expand_dims(img_array, axis=-1)

                img_array = img_array.astype(np.float32)
                img_array = img_array / 255.

                # data augmentation
                img_array = tf.keras.preprocessing.image.random_rotation(img_array, 0.2)
                img_array = tf.keras.preprocessing.image.random_shift(img_array, 0.1, 0.1)
                img_array = tf.keras.preprocessing.image.random_shear(img_array, 0.1)
                img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.7,1))

                img_array = np.array(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                if(i == 0):
                    image_batch = img_array
                else:
                    image_batch = np.concatenate((image_batch, img_array), axis=0)

            loss, optimizer = train_step(image_batch, optimizer)

            # print("training Loss: ", loss)

        if(epoch == 100):
            learning_rate = learning_rate * 0.1
            optimizer = tf.keras.optimizers.Adam(learning_rate)

        if(epoch == 150):
            learning_rate = learning_rate * 0.1
            optimizer = tf.keras.optimizers.Adam(learning_rate)

        if(epoch % 10 == 0):
            print("saveing model")
            ckpt_manager.save()

epochs = 180 + 1
BATCH_SIZE = 128
train(train_images, test_images, epochs, BATCH_SIZE)
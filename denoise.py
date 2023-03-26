"""
denoise.py

This module contains functions to denoise fluorescence microscopy images.
"""
import os
import numpy as np
from data_pipe import load_fmd_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_HEIGHT, IMG_WIDTH = 900, 900
TEST_SIZE = 0.2
EPOCHS = 10


def get_model():
    x = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Encoder
    e_conv1 = tf.keras.layers.Conv2D(64, (3, 3),
                                     activation='relu', padding='same')(x)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(e_conv1)
    batchnorm_1 = tf.keras.layers.BatchNormalization()(pool1)

    e_conv2 = tf.keras.layers.Conv2D(32, (3, 3),
                                     activation='relu', padding='same')(batchnorm_1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(e_conv2)
    batchnorm_2 = tf.keras.layers.BatchNormalization()(pool2)

    e_conv3 = tf.keras.layers.Conv2D(16, (3, 3),
                                     activation='relu', padding='same')(batchnorm_2)

    h = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(e_conv3)

    # Decoder
    d_conv1 = tf.keras.layers.Conv2D(64, (3, 3),
                                     activation='relu', padding='same')(h)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(d_conv1)
    d_conv2 = tf.keras.layers.Conv2D(32, (3, 3),
                                     activation='relu', padding='same')(up1)
    up2 = tf.keras.layers.UpSampling2D((2, 2))(d_conv2)
    d_conv3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(up2)
    up3 = tf.keras.layers.UpSampling2D((2, 2))(d_conv3)

    r = tf.keras.layers.Conv2D(3, (3, 3),
                               activation='sigmoid', padding='same')(up3)

    model = tf.keras.Model(x, r)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    return model


def train_denoise_model(noisy_images, denoised_images, filename=None):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(noisy_images), np.array(denoised_images), test_size=TEST_SIZE
    )

    # Get compiled model
    denoise_model = get_model()

    # Fit model
    history = denoise_model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=2
    )

    if filename is not None:
        denoise_model.save(filename)
        print(f"Model saved to {filename}")


noisy, denoised = load_fmd_data("/Volumes/T7/FMD_dataset")
train_denoise_model(noisy, denoised, filename='trained_models/denoise_model')

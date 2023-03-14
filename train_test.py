import sys
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_pipe import *

IMG_WIDTH = 128
IMG_HEIGHT = 128
TEST_SIZE = 0.2
EPOCHS = 10

def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python train_test.py data_directory [model.h5]")
    
    # Get image arrays and labels for all images
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Create a convolutional neural network
    model = tf.keras.models.Sequential(
        [
            # Rescale pixel values
            tf.keras.layers.Rescaling(1./255),

            # Perform convolution and pooling three times
            tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

            # Flatten units
            tf.keras.layers.Flatten(),

            # Add hidden layers with dropout
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),

            # Add an output layer with output units for all 2 categories
            tf.keras.layers.Dense(2, activation="softmax")
        ]
    )

    # Train neural net
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()

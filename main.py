import sys
import os
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

IMG_WIDTH = 256
IMG_HEIGHT = 256
TEST_SIZE = 0.4
EPOCHS = 10

def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python main.py data_directory classifications [model.h5]")
    
    # Get image arrays and labels for all images
    images, labels = load_data(sys.argv[1], sys.argv[2])

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

    if len(sys.argv) == 4:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir, classficiations):
    images = []
    labels = []

    with open(os.path.join(classficiations), 'r') as f:
        for line in f:
            print("parsing line ", line)

            # separating parsed line
            data = line.split(';')
            
            # parse filename
            filename = data[0]

            # parse label
            label = data[-1]
            labels.append(label)
            
            # parse image as ndarray
            im = cv2.imread(os.path.join(data_dir, filename))

            # resize image
            resizeIM = cv2.resize(im, (IMG_HEIGHT, IMG_WIDTH))
            images.append(resizeIM)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Create a convolutional neural network
    model = tf.keras.models.Sequential(
        [
            # convolutional layer
            tf.keras.layers.Conv2D(
                32, (3,3), activation="relu", input_shape=(IMG_HEIGHT,IMG_WIDTH,3)
            ),
            # Max-pooling layer
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

            # Flatten units
            tf.keras.layers.Flatten(),

            # Add a hidden layer with dropout
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),

            # Add an output layer with output units for all 43 categories
            tf.keras.layers.Dense(2, activation="softmax")
        ]
    )

    print(model.summary())

    # Train neural net
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()

import sys
import numpy as np

from datetime import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt

from data_pipe import *

IMG_WIDTH = 128
IMG_HEIGHT = 128
TEST_SIZE = 0.2
EPOCHS = 10

def main():
    if len(sys.argv) not in [2, 3]:
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
    history = model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    test = model.evaluate(x_test,  y_test, verbose=2)

    # Plot assessment of model
    plot_model(history, test, sys.argv[1].split('/')[-1])

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

def plot_model(history, test, model, dir=""):
    """
    This function plots the accuracy and loss of a model with respect
    to training epochs.
    """
    # epoch list
    nb_epochs = list(range(1, EPOCHS+1))

    # parse data
    train_accuracy = history.history["accuracy"]
    train_loss = history.history["loss"]

    test_accuracy = [test[1]] * EPOCHS
    test_loss = [test[0]] * EPOCHS

    # matplotlib settings
    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    # big plots please...
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5))

    # add padding
    fig.tight_layout(pad=3.0)

    # plot data
    ax1.scatter(nb_epochs, train_accuracy, color="blue", label='training accuracy')
    ax1.plot(nb_epochs, test_accuracy, color='black', linestyle='dashed', label='test')

    ax2.scatter(nb_epochs, train_loss, color = "red", label='training loss')
    ax2.plot(nb_epochs, test_loss, color='black', linestyle='dashed')

    # set tick params
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    # set title
    ax1.set_title("Accuracy of model")
    ax2.set_title("Loss of model")

    # label axis
    ax1.set_xlabel("Training Epochs", fontsize=15)
    ax1.set_ylabel("Accuracy", fontsize=15)

    ax2.set_xlabel("Training Epochs", fontsize=15)
    ax2.set_ylabel("Loss", fontsize=15)

    # plot legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper right", frameon=False)

    # set end result as text
    ax2.text(1.3 * EPOCHS, 0.5 * max(train_loss), f"Result after training: \naccuracy = {round(test[1]*100)}%")

    # Adjusting the sub-plots
    plt.subplots_adjust(right=0.7)

    # save figure
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join("accuracy_plots", model + "_" + date + ".png")
    fig.savefig(filename, bbox_inches='tight', dpi=400)


if __name__ == "__main__":
    main()

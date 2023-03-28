from evaluate import evaluate
from data_pipe import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split, KFold
from sklearn.calibration import IsotonicRegression as IR
import argparse
from pathvalidate.argparse import validate_filepath_arg
import sys
import numpy as np
import math
import pandas as pd

from datetime import datetime

# ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # noqa
from tensorflow.keras.optimizers.experimental import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TEST_SIZE = 0.2
EPOCHS = 10


def main():
    """
    This function is the entry point for the train_test.py script. It parses the
    command line arguments to determine whether the user wants to train a model
    or test a trained model.
    """

    parser = argparse.ArgumentParser(
        description="Train or test a neural net."
    )
    parser.add_argument("-f", "--function",  type=str,
                        help="desired functionality of module", required=True,
                        choices=["train", "test", "ece"])

    parser.add_argument("-c", "--classfile", type=validate_filepath_arg,
                        required=not ("-p" in sys.argv or "--path" in sys.argv))

    parser.add_argument("-p", "--path", type=check_dir_path,
                        required=not ("-c" in sys.argv or "--classfile" in sys.argv) or "test" in sys.argv or "ece" in sys.argv)

    parser.add_argument("-m", "--model", type=check_dir_path,
                        required="test" in sys.argv or "ece" in sys.argv)

    args = vars(parser.parse_args())

    if args["function"] == "train":
        # Get image arrays and labels for all images
        if args["classfile"] is not None:
            images, labels = load_data_from_classfile(
                class_file=args["classfile"])
        else:
            images, labels = load_data_from_dir(data_dir=args["path"])

        train_model(labels, images, filename=args["model"] or None)

    elif args["function"] == "test":
        accuracy, true_p, true_n = test_model(
            model_path=args["model"], data_dir=args["classfile"] or args["path"]
        )
        print(
            f"accuracy: {accuracy}, true positive: {true_p}, true_negative: {true_n}")

    elif args["function"] == "ece":
        get_model_ece(model=args["model"],
                      data_dir=args["classfile"] or args["path"])


def check_dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def train_model(labels, images, filename=None):
    """
    This function trains a convolutional neural network on a set of
    labeled images. It takes as input the image labels and pixel arrays
    as well as an optional filename to save the trained model.
    """
    # # Split data into training and testing sets
    # labels = tf.keras.utils.to_categorical(labels, num_classes=2)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # print summary of the model
    model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))
    print(model.summary(), "\n\n")

    # Fit and evaluate model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS,
                        validation_data=(x_test, y_test))

    # Plot assessment of model
    plot_model(history)

    if filename is not None:
        model.save(filename)
        print(f"Model saved to {filename}.")


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    ])

    # Create a convolutional neural network
    model = tf.keras.models.Sequential(
        [
            # Normalize data
            tf.keras.layers.Rescaling(1./255),

            # augment data
            data_augmentation,

            # Perform convolution and pooling five times
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten units
            tf.keras.layers.Flatten(),

            # Add hidden layers with dropout
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.4),

            # Add an output layer with output units for all 2 categories
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
    )

    # Train neural net
    model.compile(
        optimizer=RMSprop(learning_rate=0.00025),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def test_model(data_dir, model_path):
    """
    This function calculates the accuracy, true positive and true negative rate, evaluated
    on all data specified in a classification text file.
    """
    # load model
    print("loading model...")
    model = tf.keras.models.load_model(model_path)

    # variables to calculate rates
    predicted_positives = 0
    actual_positives = 0
    predicted_negatives = 0
    actual_negatives = 0

    mistakes = []
    low_confidence = []

    filenames, labels = load_data_df(data_dir)

    for filename, label in zip(filenames, labels):
        result = evaluate(filename, model)
        prediction = result[0]
        confidence = result[1]

        if confidence < 0.55:
            low_confidence.append(filename)

        if label == "infected":
            actual_positives += 1
            if prediction == 1:
                predicted_positives += 1
            else:
                mistakes.append(filename)

        else:
            actual_negatives += 1
            if prediction == 0:
                predicted_negatives += 1
            else:
                mistakes.append(filename)

    accuracy = (predicted_positives + predicted_negatives) / \
        (actual_positives + actual_negatives)
    true_positive = predicted_positives / actual_positives
    true_negative = predicted_negatives / actual_negatives

    print(f"\n\nSee low confidence:")
    for low in low_confidence:
        print(low)

    print(f"\n\nSee mistakes:")
    for mistake in mistakes:
        print(mistake)

    return accuracy, true_positive, true_negative


def get_model_ece(model, data_dir):
    """
    This function calculates the expected calibration error (ECE).
    """
    # load model
    print("loading model...")
    model = tf.keras.models.load_model(model)

    inf_occurences = [0] * 10
    not_inf_occurences = [0] * 10

    filenames, labels = load_data_df(data_dir)

    print(filenames)

    for image in filenames:
        # predict state of cell culture image
        prediction = evaluate(image, model)
        result = prediction[0]
        confidence = prediction[1]

        # calculate index in list
        i = math.floor(10 * confidence) - 1

        if result:
            inf_occurences[i] += 1
        else:
            not_inf_occurences[i] += 1

    # calculate probabilities
    inf_fractions = [i / sum(inf_occurences) for i in inf_occurences]
    not_inf_fractions = [i / sum(not_inf_occurences)
                         for i in not_inf_occurences]

    # plot results
    plot_ece(inf_fractions, not_inf_fractions)


def plot_ece(inf_fractions, not_inf_fractions):
    probs = list(np.arange(0.1, 1.1, 0.1))

    # matplotlib settings
    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    # big plots please...
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5))

    # add padding
    fig.tight_layout(pad=3.0)

    # plot data
    ax1.plot(probs, probs, linestyle="dashed", linewidth=2, color='black')
    ax1.plot(probs, inf_fractions, color='blue', marker='o')

    ax2.plot(probs, probs, linestyle="dashed", linewidth=2, color='black')
    ax2.plot(probs, not_inf_fractions, color='blue', marker='o')

    # set tick params
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    # set title
    ax1.set_title("ECE: positive")
    ax2.set_title("ECE: negative")

    # label axis
    ax1.set_xlabel("probabilities", fontsize=15)
    ax1.set_ylabel("fraction of positive", fontsize=15)

    ax2.set_xlabel("probabilities", fontsize=15)
    ax2.set_ylabel("fraction of negative", fontsize=15)

    # plot legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper right", frameon=False)

    # Adjusting the sub-plots
    plt.subplots_adjust(right=0.7)

    # save figure
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join("ece_plots", date + ".png")
    fig.savefig(filename, bbox_inches='tight', dpi=400)


def plot_model(history, dir=""):
    """
    This function plots the accuracy and loss of a model with respect
    to training epochs.
    """
    # epoch list
    nb_epochs = list(range(1, EPOCHS+1))

    # parse data
    train_accuracy = history.history["accuracy"]
    train_loss = history.history["loss"]

    test_accuracy = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    # matplotlib settings
    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    # big plots please...
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5))

    # add padding
    fig.tight_layout(pad=3.0)

    # plot data
    ax1.plot(nb_epochs, train_accuracy, color="blue",
             marker='o', label='training')
    ax1.plot(nb_epochs, test_accuracy, color='red',
             marker='o', label='validation')

    ax2.plot(nb_epochs, train_loss, marker='o', color="blue")
    ax2.plot(nb_epochs, test_loss, marker='o', color='red')

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
    ax2.text(1.3 * EPOCHS, 0.5 * max(train_loss),
             f"Result after training: \naccuracy = {round(test_accuracy[-1]*100, 2)}%")

    # Adjusting the sub-plots
    plt.subplots_adjust(right=0.7)

    # save figure
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join("plots", "accuracy_plots",
                            "model_" + date + ".png")
    fig.savefig(filename, bbox_inches='tight', dpi=400)


def plot_model_kf_vc(accuracy, loss, model, dir=""):
    """
    This function plots the accuracy and loss of a model with respect
    to training epochs.

    Not implemented
    """
    # epoch list
    nb_epochs = list(range(1, EPOCHS+1))

    # matplotlib settings
    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    # big plots please...
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5))

    # add padding
    fig.tight_layout(pad=3.0)

    # plot data
    ax1.scatter(nb_epochs, accuracy, color="blue", label='training accuracy')
    ax1.plot(nb_epochs, loss, color='black', linestyle='dashed', label='test')

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

    # # set end result as text
    # ax2.text(1.3 * EPOCHS, 0.5 * max(train_loss), f"Result after training: \naccuracy = {round(test[1]*100, 2)}%")

    # Adjusting the sub-plots
    plt.subplots_adjust(right=0.7)

    # save figure
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join("accuracy_plots", model + "_" + date + ".png")
    fig.savefig(filename, bbox_inches='tight', dpi=400)


if __name__ == "__main__":
    main()

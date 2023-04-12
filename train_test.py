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
EPOCHS = 32


def main():
    """
    This function is the entry point for the train_test.py script. It parses the
    command line arguments to determine whether the user wants to train a model
    or test a trained model.
    """
    # create parser object
    parser = argparse.ArgumentParser(
        description="Train or test a neural net."
    )

    # add argument to specify desired function
    parser.add_argument("-f", "--function",  type=str,
                        help="desired functionality of module", required=True,
                        choices=["train", "test", "visualize", "ece"])

    # add argument to specify which classification file to choose
    parser.add_argument("-c", "--classfile", type=validate_filepath_arg,
                        required=not ("-p" in sys.argv or "--path" in sys.argv))

    # add argument to specify which path (containing images) to choose
    parser.add_argument("-p", "--path", type=check_dir_path,
                        required=not ("-c" in sys.argv or "--classfile" in sys.argv))

    parser.add_argument("-i", "--image_path")

    # specify path to trained modle
    parser.add_argument("-m", "--model", type=check_dir_path,
                        required="test" in sys.argv or "ece" in sys.argv)

    # optionial denoising model
    parser.add_argument("-d", "--denoiser", type=check_dir_path)

    parser.add_argument("-l", "--limit", type=int)

    # convert parsed arguments to dictionary
    args = vars(parser.parse_args())

    # call functions according to specified arguments
    if args["function"] == "train":
        # Get image arrays and labels for all images
        if args["classfile"] is not None:
            images, labels = load_data_from_classfile(
                class_file=args["classfile"])
        else:
            images, labels = load_data_from_dir(data_dir=args["path"])

        train_model(labels, images, filename=args["model"] or None)

    elif args["function"] == "test":
        # distinction between classficiation file or data directory as input
        if args["classfile"] is not None:
            accuracy, true_p, true_n = test_model(
                model_path=args["model"], classfile=args["classfile"], limit=args["limit"] or np.inf
            )
        elif args["path"] is not None:
            accuracy, true_p, true_n = test_model(
                model_path=args["model"], data_dir=args["path"], limit=args["limit"] or np.inf
            )

        print(
            f"accuracy: {accuracy}, true positive: {true_p}, true_negative: {true_n}")

    elif args["function"] == "ece":
        get_model_ece(model=args["model"],
                      data_dir=args["classfile"] or args["path"], limit=args["limit"] or np.inf)


def check_dir_path(path):
    """
    This function is used by argparse to check if an argument is a directory.
    """
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
    earlystopper = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filename, verbose=1,
                                                      save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS,
                        validation_data=(x_test, y_test),
                        callbacks=[earlystopper, checkpointer])

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
        tf.keras.layers.RandomFlip("horizontal_and_vertical")
    ])

    # Create a convolutional neural network
    model = tf.keras.models.Sequential(
        [
            # Normalize data
            tf.keras.layers.Rescaling(1./255),

            # augment data
            data_augmentation,

            # Perform convolution and pooling five times
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten units
            tf.keras.layers.Flatten(),

            # Add hidden layers with dropout
            tf.keras.layers.Dense(512, activation="relu"),

            # Add an output layer with output units for all 2 categories
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
    )

    # Train neural net
    model.compile(
        optimizer=RMSprop(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def test_model(classfile=None, data_dir=None, model_path=None, limit=np.inf):
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

    mistakes = {}
    low_confidence = {}

    # if classification file specified
    if classfile is not None:
        filepaths, labels = load_data_df_from_class(classfile, limit=limit)
    else:
        filepaths, labels = load_data_df_from_dir(data_dir, limit=limit)

    for filepath, label in zip(filepaths, labels):
        result = evaluate(filepath, model=model)
        prediction = result[0]
        confidence = result[1]

        if confidence < 0.55:
            low_confidence[filepath] = prediction

        if label == "infected":
            actual_positives += 1
            if prediction == 1:
                predicted_positives += 1
            else:
                mistakes[filepath] = prediction

        else:
            actual_negatives += 1
            if prediction == 0:
                predicted_negatives += 1
            else:
                mistakes[filepath] = prediction

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


def get_model_ece(model, data_dir, limit=np.inf):
    """
    This function calculates the expected calibration error (ECE).
    """
    # load model
    print("loading model...")
    model = tf.keras.models.load_model(model)

    inf_occurences = [0] * 10
    not_inf_occurences = [0] * 10

    if os.path.isdir(data_dir):
        filenames, labels = load_data_df_from_dir(data_dir, limit=limit)
    else:
        filenames, labels = load_data_df_from_class(data_dir, limit=limit)

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
    filename = os.path.join("plots", "ece_plots", date + ".png")
    fig.savefig(filename, bbox_inches='tight', dpi=400)


def plot_model(history, dir=""):
    """
    This function plots the accuracy and loss of a model with respect
    to training epochs.
    """
    # epoch list
    nb_epochs = len(list(history.history['val_accuracy']))
    epochs = list(range(1, nb_epochs + 1))

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
    ax1.plot(epochs, train_accuracy, color="blue",
             marker='o', label='training')
    ax1.plot(epochs, test_accuracy, color='red',
             marker='o', label='validation')

    ax2.plot(epochs, train_loss, marker='o', color="blue")
    ax2.plot(epochs, test_loss, marker='o', color='red')

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
    ax2.text(1.3 * nb_epochs, 0.5 * max(train_loss),
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


def viszalize_model(image, model):
    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = tf.keras.Model(input=model.input,
                                         outputs=successive_outputs)
    successive_feature_maps = visualization_model.predict(x)
    layer_names = [layer.name for layer in model.layers]

    # plot everything
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):

        if len(feature_map.shape) == 4:  # if it is a conv or pooling layer
            n_features = feature_map.shape[-1]  # n features
            size = feature_map.shape[1]  # shape

            # create a grid to display the data
            display_grid = np.zeros((size, size * n_features))

            # some post-processing
            for i in range(n_features):
                image = feature_map[0, :, :, i]
                image -= image.mean()
                image /= image.std()
                image *= 64
                image += 128
                image = np.clip(image, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = image

                # show the chart
                scale = 20. / n_features
                plt.figure(figsize=(scale * n_features, scale))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')


if __name__ == "__main__":
    main()

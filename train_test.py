import sys
import numpy as np

from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers.experimental import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, KFold

import matplotlib as mpl
import matplotlib.pyplot as plt

from data_pipe import *
from evaluate import evaluate

IMG_WIDTH = 256
IMG_HEIGHT = 256
TEST_SIZE = 0.2
EPOCHS = 10

def main():
    commands = {"-help":None,
                "-train":("data_directory", "[model.h5]"),
                "-test":("model", "data_directory")}

    if len(sys.argv) >= 5:
        sys.exit("Usage: python train_test.py -train data_directory [model.h5]")
    
    elif sys.argv[1] == "-train":
        # Get image arrays and labels for all images
        images, labels = load_data(sys.argv[2])
        filename = sys.argv[3] if len(sys.argv) == 4 else None

        train_model(labels, images, filename=filename)
        
        # train_model(images, labels, filename)
        # accuracy, loss = train_kfold_vc(train_data, sys.argv[1], save_dir='kf_vc')
        # plot_model_kf_vc(accuracy, loss, "k-fold cross validation")
        # train_data = load_data_df(sys.argv[1])
    
    elif sys.argv[1] == "-test":
        model = sys.argv[2]
        data_dir = sys.argv[3]

        accuracy, true_p, true_n = test_model(model, data_dir)

        print(f"accuracy: {accuracy}, true positive: {true_p}, true_negative: {true_n}")


def train_model(labels, images, filename=None):
    """
    This function trains a model simply by splitting all loaded data into a training and
    testing set. The resulting training accuracies and losses as well as the end result 
    are plotted.
    """
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

    if filename is not None:
        model.save(filename)
        print(f"Model saved to {filename}.")

def train_kfold_vc(train_data, data_dir, save_dir=''):
    """
    
    """
    validation_accuracy = []
    validation_loss =  []

    fold_var = 1

    # shuffle train_data df
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    # get dataframe of filenames and labels
    Y = train_data[["labels"]]

    kf = KFold(n_splits=5)

    idg = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.3,
                             fill_mode='nearest',
                             horizontal_flip=True)
    

    
    for train_index, val_index in kf.split(np.zeros(len(Y)), Y):

        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]

        print(training_data)
        print(validation_data)

        train_data_generator = idg.flow_from_dataframe(training_data,
                                                       x_col='filenames', y_col='labels',
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary', shuffle=True,
                                                       validate_filenames=False)


        valid_data_generator = idg.flow_from_dataframe(validation_data,
                                                       x_col='filenames', y_col='labels',
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary', shuffle=True,
                                                       validate_filenames=False)

        
        model = get_model()

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{save_dir}/model_{fold_var}.h5",
                                                        monitor='val_accuracy', verbose=1,
                                                        save_best_only=True, mode='max')
        
        callbacks_list = [checkpoint]

        history = model.fit(train_data_generator,
                            epochs=EPOCHS,
                            callbacks=callbacks_list,
                            validation_data=valid_data_generator)
        
        model.load_weights(f"{save_dir}/model_{fold_var}.h5")
        
        results = model.evaluate(valid_data_generator)
        results = dict(zip(model.metrics_names,results))

        validation_accuracy.append(results['accuracy'])
        validation_accuracy.append(results['loss'])

        tf.keras.backend.clear_session()

        fold_var += 1
    
    return validation_accuracy, validation_loss


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
            tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

            # Flatten units
            tf.keras.layers.Flatten(),

            # Add hidden layers with dropout
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.3),

            # Add an output layer with output units for all 2 categories
            tf.keras.layers.Dense(2, activation="sigmoid")
        ]
    )

    # Train neural net
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def test_model(model, data_dir):
    """
    This function calculates the accuracy, true positive and true negative rate, evaluated
    on data in 'data_dir'.
    """
    # load model
    model = tf.keras.models.load_model(model)

    # variables to calculate rates
    predicted_positives = 0
    actual_positives = 0
    predicted_negatives = 0
    actual_negatives = 0

    mistakes = []

    filenames, labels = load_data_df(data_dir)

    for filename, label in zip(filenames, labels):
        print(filename, label)
        prediction = evaluate(filename, model)[0]

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
    
    accuracy = (predicted_positives + predicted_negatives) / (actual_positives + actual_negatives)
    true_positive = predicted_positives / actual_positives
    true_negative = predicted_negatives / actual_negatives


    for mistake in mistakes:
        print(mistake)

    return accuracy, true_positive, true_negative


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
    ax2.text(1.3 * EPOCHS, 0.5 * max(train_loss), f"Result after training: \naccuracy = {round(test[1]*100, 2)}%")

    # Adjusting the sub-plots
    plt.subplots_adjust(right=0.7)

    # save figure
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join("accuracy_plots", model + "_" + date + ".png")
    fig.savefig(filename, bbox_inches='tight', dpi=400)

def plot_model_kf_vc(accuracy, loss, model, dir=""):
    """
    This function plots the accuracy and loss of a model with respect
    to training epochs.
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

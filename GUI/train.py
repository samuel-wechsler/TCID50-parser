import os
import sys

import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from filehandling import load_data_from_df

# Import tensorflow libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # noqa
from tensorflow.keras.optimizers.experimental import RMSprop  # type: ignore

from params import TrainParams

IMG_HEIGHT, IMG_WIDTH = 256, 256


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--train-data-file', type=str, required=True,
                        help='path to training data file')
    parser.add_argument('--model-save-file', type=str, required=True,
                        help='path to save model'),
    parser.add_argument('--train-mode', type=str, default="replace"),
    parser.add_argument('--epochs', type=int, default=12,
                        help='number of epochs')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='validation split')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size'),
    parser.add_argument('--dropout', type=float, default=0.2),
    parser.add_argument('--rotation', type=float, default=0.785,
                        help='rotation angle')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer')
    parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy'],
                        help='metrics')
    parser.add_argument('--horiz-flip', type=bool)
    parser.add_argument('--vert-flip', type=bool)

    args = parser.parse_args()

    params = TrainParams(train_data_file=args.train_data_file,
                         model_save_file=args.model_save_file,
                         epochs=args.epochs,
                         validation_split=args.validation_split,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size,
                         dropout=args.dropout,
                         rotation=args.rotation,
                         optimizer=args.optimizer,
                         metrics=args.metrics,
                         horiz_flip=args.horiz_flip,
                         vert_flip=args.vert_flip)

    # train the model
    train = Train(params)
    train.train_model()


class Train:
    def __init__(self, train_params):
        self.train_params = train_params

    def get_class_dataframe(self):
        df = pd.read_csv(self.train_params.train_data_file, sep=";",
                         header=0, names=["images", "labels"])
        df["labels"] = df["labels"].astype(str)

        return df.to_numpy()

    def compute_class_weights(self, labels):
        """
        Compute class weights based on the distribution of classes in the labels.
        """
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(labels), y=labels)

        class_weights = {i: weight for i, weight in zip(
            np.unique(labels), class_weights)}

        return class_weights

    def check_class_imbalance(self):
        # Read the CSV file containing the training data
        df = self.get_class_dataframe()

        # Get the counts for each class
        counts = df['labels'].value_counts()

        # Calculate the class imbalance ratio
        imbalance_ratio = counts.min() / counts.max()

        # Print the class imbalance ratio
        print(f"Class imbalance ratio: {imbalance_ratio:0.2f}")

    def get_optimizer(self):
        if self.train_params.optimizer == "RMSprop":
            return RMSprop(learning_rate=self.train_params.learning_rate)

        if self.train_params.optimizer == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.train_params.learning_rate)

        if self.train_params.optimizer == "Adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=self.train_params.learning_rate)

    def get_model(self, pretrained_model=None):
        """
        Returns a compiled convolutional neural network model. Assume that the
        `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
        The output layer should have `NUM_CATEGORIES` units, one for each category.
        """

        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical" if (self.train_params.horiz_flip and self.train_params.vert_flip)
                                       else "horizontal" if self.train_params.horiz_flip else "vertical" if self.train_params.vert_flip else None),
        ])

        # Create a convolutional neural network
        model = tf.keras.models.Sequential()

        if pretrained_model is not None:
            model.add(pretrained_model)

        # Normalize input data
        model.add(tf.keras.layers.Rescaling(1./255))

        # Data augmentation
        model.add(data_augmentation)

        # Perform convolution and pooling five times
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Dropout(self.train_params.dropout))

        # Flatten units
        model.add(tf.keras.layers.Flatten())

        # Add hidden layers with dropout
        model.add(tf.keras.layers.Dense(512, activation="relu"))

        # Add an output layer with output units for all 2 categories
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Train neural net
        model.compile(
            optimizer=self.get_optimizer(),
            loss="binary_crossentropy",
            metrics=self.train_params.metrics
        )

        return model

    def save_model_json(self, model):
        """
        save model architecture as json file
        """
        model_json = model.to_json()

        # get save path from save_file
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(self.train_params.model_save_file)
                            ) or os.path.dirname(os.getcwd()),
            os.path.dirname(self.train_params.model_save_file),
            "model_architectures",
            os.path.splitext(os.path.basename(
                self.train_params.model_save_file))[0] + ".json"
        )

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path, "w") as json_file:
            json_file.write(model_json)

    def train_model(self):
        """
        train the model
        """
        images, labels = load_data_from_df(self.get_class_dataframe(),
                                           img_size=IMG_HEIGHT)

        # split data into training and validation sets
        train_images, test_images, train_labels, test_labels = train_test_split(
            np.array(images), np.array(labels), test_size=self.train_params.validation_split
        )

        model = None

        if self.train_params.train_mode == "transfer":
            model = tf.keras.models.load_model(
                self.train_params.model_save_file
            )

            for layer in model.layers:
                layer.trainable = False

        model = self.get_model(pretrained_model=model)

        # print summary of the model
        model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))
        print(model.summary(), f"\n")

        # Fit and evaluate model on training data
        earlystopper = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(self.train_params.model_save_file,
                                                          verbose=1,
                                                          save_best_only=True)
        cb = [earlystopper, checkpointer]

        print(
            f"training model... with class weights {self.compute_class_weights(labels)}\n")

        history = model.fit(train_images, train_labels,
                            validation_data=(test_images, test_labels),
                            epochs=self.train_params.epochs,
                            callbacks=cb,
                            verbose=1,
                            class_weight=self.compute_class_weights(
                                train_labels
                            )
                            )

        # save model architecture as json file
        self.save_model_json(model)


if __name__ == "__main__":
    main()

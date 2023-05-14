import os
import sys

import pandas as pd
import argparse

# Import tensorflow libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # noqa
from tensorflow.keras.optimizers.experimental import RMSprop  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

from params import TrainParams

IMG_HEIGHT, IMG_WIDTH = 256, 256


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--train-data-file', type=str, required=True,
                        help='path to training data file')
    parser.add_argument('--model-save-file', type=str, required=True,
                        help='path to save model')
    parser.add_argument('--epochs', type=int, default=12,
                        help='number of epochs')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='validation split')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--rotation', type=float, default=0.785,
                        help='rotation angle')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer')
    parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy'],
                        help='metrics')

    args = parser.parse_args()

    params = TrainParams(train_data_file=args.train_data_file,
                         model_save_file=args.model_save_file,
                         epochs=args.epochs,
                         validation_split=args.validation_split,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size,
                         rotation=args.rotation,
                         optimizer=args.optimizer,
                         metrics=args.metrics)

    # train the model
    train = Train(params)
    train.train_model()


class Train:
    def __init__(self, train_params):
        self.train_params = train_params

    def get_class_dataframe(self):
        df = pd.read_csv(self.train_params.train_data_file, sep=";",
                         header=0, names=["file", "label"])
        df["label"] = df["label"].astype(str)
        return df

    def get_optimizer(self):
        if self.train_params.optimizer == "RMSprop":
            return RMSprop(learning_rate=self.train_params.learning_rate)

        if self.train_params.optimizer == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.train_params.learning_rate)

        if self.train_params.optimizer == "Adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=self.train_params.learning_rate)

    def get_model(self):
        """
        Returns a compiled convolutional neural network model. Assume that the
        `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
        The output layer should have `NUM_CATEGORIES` units, one for each category.
        """

        # Create a convolutional neural network
        model = tf.keras.models.Sequential(
            [
                # Perform convolution and pooling five times
                tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                tf.keras.layers.Dropout(0.2),

                # Flatten units
                tf.keras.layers.Flatten(),

                # Add hidden layers with dropout
                tf.keras.layers.Dense(128, activation="relu"),

                # Add an output layer with output units for all 2 categories
                tf.keras.layers.Dense(1, activation="sigmoid")
            ]
        )

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

        # Define the ImageDataGenerator
        datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=self.train_params.rotation,
                                     horizontal_flip=bool(
                                         self.train_params.rotation),
                                     vertical_flip=bool(
                                         self.train_params.rotation),
                                     validation_split=self.train_params.validation_split)

        df = self.get_class_dataframe()

        # Create the ImageDataGenerator from the DataFrame
        train_generator = datagen.flow_from_dataframe(
            dataframe=df,
            x_col='file',
            y_col='label',
            target_size=(IMG_HEIGHT, IMG_HEIGHT),  # Set the target image size
            batch_size=self.train_params.batch_size,
            class_mode='binary',
            subset='training'
        )

        # Create a separate ImageDataGenerator for the validation set
        validation_generator = datagen.flow_from_dataframe(
            dataframe=df,
            x_col='file',
            y_col='label',
            target_size=(IMG_HEIGHT, IMG_HEIGHT),
            batch_size=self.train_params.batch_size,
            class_mode='binary',
            subset='validation'
        )

        model = self.get_model()

        # print summary of the model
        model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))
        print(model.summary(), f"\n")

        # Fit and evaluate model on training data
        earlystopper = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(self.train_params.model_save_file,
                                                          verbose=1,
                                                          save_best_only=True)
        cb = [earlystopper, checkpointer]

        history = model.fit(train_generator,
                            validation_data=validation_generator,
                            epochs=self.train_params.epochs,
                            callbacks=cb,
                            verbose=1)

        # save model architecture as json file
        self.save_model_json(model)


if __name__ == "__main__":
    main()

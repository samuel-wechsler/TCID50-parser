import numpy as np


class AutomateConfig:
    """ Configuration object for ClassifyThread and ClassifyPlates """

    def __init__(self, plates_dir, model_path, img_paths, classified, row_range, col_range, serial_dilution, initial_dilution, particle_to_pfu, classify_completed):
        """
        plates_dir: directory containing plate images
        model_path: path to model file
        imgs: list of image files (to be classified, some images in dir may have been filtered out)
        classified: dict of classified images with bool values
        """
        self.plates_dir = plates_dir
        self.model_path = model_path
        self.model = None
        self.img_paths = img_paths
        self.classified = classified
        self.row_range = row_range
        self.col_range = col_range
        self.serial_dilution = serial_dilution
        self.initial_dilution = initial_dilution
        self.particle_to_pfu = particle_to_pfu
        self.classify_completed = classify_completed


class TrainParams:
    """ Configuration object to specfiy training parameters """

    def __init__(self, train_data_file, model_save_file, train_mode="replace", epochs=12, validation_split=0.2, learning_rate=0.001, batch_size=32, dropout=0.2, rotation=0.2, optimizer="Adam", metrics=["accuracy"], horiz_flip=True, vert_flip=True):
        """
        train_data_file: path to training data file
        model_save_file: path to save model
        epochs: number of epochs
        learning_rate: learning rate
        batch_size: batch size
        loss: loss function
        optimizer: optimizer
        metrics: metrics
        """
        self.train_data_file = train_data_file
        self.model_save_file = model_save_file
        self.train_mode = train_mode
        self.epochs = epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.rotation = rotation
        self.optimizer = optimizer
        self.metrics = metrics
        self.horiz_flip = horiz_flip
        self.vert_flip = vert_flip

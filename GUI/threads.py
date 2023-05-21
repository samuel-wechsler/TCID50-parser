import threading

import subprocess
import platform
import os
from datetime import datetime

from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal


print("loading tensorflow...")  # noqa
start = datetime.now()  # noqa
from classify import ClassifyPlates
end = datetime.now()  # noqa
print("tensorflow loaded in {} seconds".format(end - start))  # noqa
from upload import Uploader
from filehandling import FileHandling


class ImgUploadThread(QThread, FileHandling):
    # Define a signal that will be emitted when the upload is complete
    upload_complete = pyqtSignal()

    def __init__(self, images, threadFinished):
        super().__init__()
        self.images = images
        self.threadFinished = threadFinished
        self.uploader = Uploader()

    def run(self):
        self.uploader.create_image_dir(
            f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        for image in self.images:
            if not self.isImageFile(image):
                image = f"Skipped: {image}"
            # Upload the image
            self.uploader.upload_image(image)
            # Emit a signal to update the progress bar and label
            self.upload_complete.emit()
        self.threadFinished()


class ModelUploadThread(QThread):
    """ Thread for uploading a model to Google Drive. """
    # Define a signal that will be emitted when the upload is complete
    upload_complete = pyqtSignal()

    def __init__(self, model_path):
        super().__init__()
        self.uploader = Uploader()
        self.model_path = model_path

    def run(self):
        # Upload the model
        self.uploader.upload_model_arch(self.model_path)
        self.upload_complete.emit()


class ClassUploadThread(QThread):
    """ Thread for uploading a model to Google Drive. """
    # Define a signal that will be emitted when the upload is complete
    upload_complete = pyqtSignal()

    def __init__(self, class_path):
        super().__init__()
        self.uploader = Uploader()
        self.class_path = class_path

    def run(self):
        # Upload the model
        self.uploader.upload_class(self.class_path)
        self.upload_complete.emit()


class TrainThread(threading.Thread):
    # TODO: consider to refactor at some point
    # since thread isn't really needed
    def __init__(self, trainParams):
        threading.Thread.__init__(self)
        self.trainParams = trainParams
        self._stop_event = threading.Event()

    def start_train_in_terminal(self):
        cwd = os.getcwd()
        program = 'train.py'
        train_args = [
            '--train-data-file', self.trainParams.train_data_file,
            '--model-save-file', self.trainParams.model_save_file,
            '--train-mode', self.trainParams.train_mode,
            '--epochs', str(self.trainParams.epochs),
            '--validation-split', str(self.trainParams.validation_split),
            '--learning-rate', str(self.trainParams.learning_rate),
            '--batch-size', str(self.trainParams.batch_size),
            '--dropout', str(self.trainParams.dropout),
            '--rotation', str(self.trainParams.rotation),
            '--optimizer', self.trainParams.optimizer,
            '--horiz-flip', str(self.trainParams.horiz_flip),
            '--vert-flip', str(self.trainParams.vert_flip)
        ]
        train_args.extend(['--metrics', *self.trainParams.metrics])

        if platform.system() == 'Windows':
            cmd = ['cmd', '/c', 'start', 'cmd', '/k', 'cd',
                   cwd, '&&', 'python', program, *train_args]
            subprocess.Popen(cmd, shell=True)
        elif platform.system() == 'Linux':
            cmd = ['gnome-terminal', '-n', '-x', 'bash', '-c',
                   'cd "{}" && python {} {}'.format(cwd, program, ' '.join(train_args))]
            subprocess.Popen(cmd, shell=False)
        elif platform.system() == 'Darwin':
            cmd = ['osascript', '-e', 'tell app "Terminal" to do script "cd {} && python {} {}"'.format(
                cwd, program, ' '.join(train_args))]
            subprocess.Popen(cmd, shell=False)

    def run(self):
        self.start_train_in_terminal()

    def stop(self):
        self._stop_event.set()


class ClassifyThread(QThread):
    """ Thread for classifiying images """
    # Define a signal that will be emitted when the upload is complete
    classify_complete = pyqtSignal(int)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        self.classify = ClassifyPlates(self.params)

        self.classify.classify_plates(thread=self)

        self.params.classify_completed()

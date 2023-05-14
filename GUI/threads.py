import threading

import subprocess
import platform
import os

from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal

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
    def __init__(self, trainParams, textBrowser, reconnectMethod):
        threading.Thread.__init__(self)
        self.trainParams = trainParams
        self.textBrowser = textBrowser
        self.reconnectMethod = reconnectMethod
        self._stop_event = threading.Event()

    def start_train_in_terminal(self):
        cwd = os.getcwd()
        program = 'model_dev/train.py'

        if platform.system() == 'Windows':
            subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', 'cd', cwd, '&&', 'python', program],
                             shell=True)
        elif platform.system() == 'Linux':
            subprocess.Popen(['gnome-terminal', '-n', '-x', 'bash', '-c', 'cd "{}" && python {}'.format(cwd, program)],
                             shell=False)
        elif platform.system() == 'Darwin':
            subprocess.Popen(['osascript', '-e', 'tell app "Terminal" to do script "cd {} && python {}"'.format(cwd, program)],
                             shell=False)

    def run(self):
        # prevent user from restarting training
        self.reconnectMethod()

        try:
            self.start_train_in_terminal()
        # # Only now importing tensorflow to avoid long loading time
        # self.textBrowser.insertPlainText(f"Importing tensorflow.\n\n")
        # from model_dev.train import Train

        # # train model on seperate thread
        # self.textBrowser.insertPlainText(f"Training model.\n\n")

        # try:
        #     self.train = Train(self.trainParams)
        #     self.train.train_model(self._stop_event)
        # except ValueError as e:
        #     self.textBrowser.insertPlainText(
        #         f"{40*'*'}\n\n VALUE ERROR: \n {e} \n\n {40*'*'}\n")
        #     self.textBrowser.insertPlainText(f"\n\nTraining aborted.\n")
        #     self.stop()
        # except Exception as e:
        #     self.textBrowser.insertPlainText(
        #         f"{40*'*'}\n\n UNKOWN ERROR: \n {e} \n\n {40*'*'}\n")
        #     self.textBrowser.insertPlainText(f"\n\nTraining aborted.\n")
        #     self.stop()
        # else:
        #     self.textBrowser.insertPlainText(f"\n\nTraining complete.\n\n")
        except Exception as e:
            raise e
        finally:
            self.reconnectMethod()

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
        from classify import ClassifyPlates

        self.classify = ClassifyPlates(self.params)

        self.classify.classify_plates(thread=self)

        self.params.classify_completed()

import os
import numpy as np

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from update import *
from upload import Uploader
from threads import *

FilterUi, QtBaseclass = uic.loadUiType("qt_files/filterdialog.ui")
UploadUi, QtBaseclass = uic.loadUiType("qt_files/uploaddlg.ui")
AutomateUi, QtBaseclass = uic.loadUiType("qt_files/automatedlg.ui")
TrainParamsUi, QtBaseclass = uic.loadUiType("qt_files/trainingparamsdlg.ui")
UpdateUi, QtBaseclass = uic.loadUiType("qt_files/updatedlg.ui")


class UpdateDlg(QDialog):
    def __init__(self):
        super(UpdateDlg, self).__init__()

        if not check_for_updates():
            pass

        self.ui = UpdateUi()
        self.ui.setupUi(self)

        self._connectWidgets()
        self.response = self.ui.comboBox.currentText()

    def _connectWidgets(self):
        self.ui.pushButton.clicked.connect(self.updateApplication)
        self.ui.comboBox.currentIndexChanged.connect(self.setResponse)

    def setResponse(self):
        self.response = self.ui.comboBox.currentText()

    def updateApplication(self):
        update = "Yes please" == self.response

        if update:
            update_application()

        self.close()


class MyDialog(QDialog):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Return:
            # ignore the event and prevent the dialog from closing
            event.ignore()
        else:
            # call the base class implementation to handle other key events
            super().keyPressEvent(event)


class FilterDialog(MyDialog):
    def __init__(self, files, callback):
        super(MyDialog, self).__init__()
        self.ui = FilterUi()
        self.ui.setupUi(self)

        self.files = files
        self.listFiles = files
        self.callback = callback

        # connect widgets
        self._connectWidget()

        # load table
        self.loadImgList()

    def _connectWidget(self):
        """
        method to connect all widgets
        """
        self.ui.buttonBox.clicked.connect(self.buttonBoxClicked)
        self.ui.lineEdit.editingFinished.connect(self.filter)
        self.ui.lineEdit.returnPressed.connect(
            lambda: self.ui.lineEdit.clearFocus()
        )

    def loadImgList(self):
        # clear previous entries from table
        self.ui.tableWidget.setRowCount(0)

        for row_nb in range(len(self.listFiles)):
            filename = os.path.splitext(
                os.path.basename(self.listFiles[row_nb]))[0]
            self.ui.tableWidget.insertRow(row_nb)
            self.ui.tableWidget.setItem(row_nb, 0, QTableWidgetItem(filename))
            self.ui.tableWidget.setItem(row_nb, 1, QTableWidgetItem(""))

    def filter(self):
        searchTerm = self.ui.lineEdit.text()
        self.listFiles = []
        for file in self.files:
            name = os.path.splitext(os.path.basename(file))[0]
            if searchTerm in name:
                self.listFiles.append(file)
        self.loadImgList()

    def buttonBoxClicked(self, option):
        option = option.text()

        if option == "OK":
            self.callback(self.listFiles)


class ImgUploadDlg(QDialog):
    def __init__(self, images):
        super(ImgUploadDlg, self).__init__()
        self.ui = UploadUi()
        self.ui.setupUi(self)

        self.images = images
        self.counter = 0
        self.nb_images = len(images)

        # start upload
        self.upload()

    def updateProgressBar(self):
        self.ui.progressBar.setValue(
            round(100 * self.counter / self.nb_images))
        self.counter += 1

    def updateFilename(self):
        self.ui.label.setText(
            f"Uploading {os.path.basename(self.images[self.counter])}")

    def uploadFinished(self):
        self.ui.label.setText("Upload succesful!")
        self.ui.progressBar.setValue(100)

    def upload(self):
        # Create an instance of the UploadThread and connect its signal to the updateProgressBar()
        # and updateFilename() slots
        self.thread = ImgUploadThread(self.images, self.uploadFinished)
        self.thread.upload_complete.connect(self.updateFilename)
        self.thread.upload_complete.connect(self.updateProgressBar)

        # Start the thread
        self.thread.start()


class ModelUploadDlg(QDialog):
    """ Dialog for uploading a model to Google Drive. """

    def __init__(self, model_arch_path):
        super(ModelUploadDlg, self).__init__()
        self.ui = UploadUi()
        self.ui.setupUi(self)

        self.model_arch_path = model_arch_path

        self.upload()

    def upload(self):
        self.thread = ModelUploadThread(self.model_arch_path)
        self.thread.upload_complete.connect(self.uploadFinished)
        self.thread.start()

    def uploadFinished(self):
        self.ui.label.setText("Upload succesful!")
        self.ui.progressBar.setValue(100)


class ClassUploadDlg(QDialog):
    """ Dialog for uploading a model to Google Drive. """

    def __init__(self, class_path):
        super(ClassUploadDlg, self).__init__()
        self.ui = UploadUi()
        self.ui.setupUi(self)

        self.class_path = class_path

        self.upload()

    def upload(self):
        self.thread = ClassUploadThread(self.class_path)
        self.thread.upload_complete.connect(self.uploadFinished)
        self.thread.start()

    def uploadFinished(self):
        self.ui.label.setText("Upload succesful!")
        self.ui.progressBar.setValue(100)


class AutomateDlg(QDialog):
    """ Dialog to configure automated classification. """
    # TODO: check that required fields are filled in

    def __init__(self):
        super(AutomateDlg, self).__init__()
        self.ui = AutomateUi()
        self.ui.setupUi(self)

        # set variables
        self.model_path = None
        self.row_range = None
        self.col_range = None
        self.serial_dilution = None
        self.initial_dilution = None
        self.part_pfu = None

        # set validators
        int_validator = QIntValidator()
        int_validator.setRange(1, 1000000000)
        self.ui.serial_dilution.setValidator(int_validator)
        self.ui.initial_dilution.setValidator(int_validator)
        self.ui.part_pfu.setValidator(int_validator)

        self._connectWidgets()

    def _connectWidgets(self):
        """ Connect all widgets to their respective methods. """
        self.ui.model_path.editingFinished.connect(self.setModelPath)
        self.ui.row_range.editingFinished.connect(self.setRowRange)
        self.ui.col_range.editingFinished.connect(self.setColRange)
        self.ui.serial_dilution.editingFinished.connect(self.setSerialDilution)
        self.ui.initial_dilution.editingFinished.connect(
            self.setInitialDilution
        )
        self.ui.part_pfu.editingFinished.connect(
            self.setParticlesToPfu
        )

    def setModelPath(self):
        # TODO: validate that model exists
        self.model_path = self.ui.model_path.text()

        if os.path.dirname(self.model_path) == "trained_models":
            self.model_path = os.path.join(
                os.path.dirname(os.getcwd()),
                "trained_models",
                os.path.basename(self.model_path)
            )

    def setRowRange(self):
        self.row_range = []

        # split string into characters
        entry = self.ui.row_range.text()
        for c in entry:
            if c.isupper() and c.isalpha():
                self.row_range.append(c)

    def setColRange(self):
        self.col_range = []

        # split string into characters
        entry = self.ui.col_range.text().replace(" ", "")
        entry = entry.split(",")
        for c in entry:
            if c.isnumeric():
                self.col_range.append(int(c))

    def setSerialDilution(self):
        self.serial_dilution = int(self.ui.serial_dilution.text())

    def setInitialDilution(self):
        self.initial_dilution = int(self.ui.initial_dilution.text())

    def setParticlesToPfu(self):
        self.part_pfu = int(self.ui.part_pfu.text())

    def getParams(self):
        return (self.model_path,
                self.row_range, self.col_range,
                self.serial_dilution, self.initial_dilution,
                self.part_pfu)

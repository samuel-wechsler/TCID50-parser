"""
GUI to classify infection state of cell cultures.
"""
import sys
import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import qdarktheme

from filehandling import FileHandling
from errorhandling import ErrorHandling
from uploadimage import Uploader

MainUI, QtBaseclass = uic.loadUiType("main.ui")
FilterUi, QtBaseclass = uic.loadUiType("filterdialog.ui")
UploadUi, QtBaseclass = uic.loadUiType("uploaddlg.ui")


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


class UploadThread(QThread):
    # Define a signal that will be emitted when the upload is complete
    upload_complete = pyqtSignal()

    def __init__(self, images, threadFinished):
        super().__init__()
        self.images = images
        self.threadFinished = threadFinished
        self.uploader = Uploader()

    def run(self):
        for image in self.images:
            # Upload the image
            self.uploader.upload_image(image)

            # Emit a signal to update the progress bar and label
            self.upload_complete.emit()
        self.threadFinished()


class UploadDlg(QDialog):
    def __init__(self, images):
        super(UploadDlg, self).__init__()
        self.ui = UploadUi()
        self.ui.setupUi(self)

        self.images = images
        self.counter = 0
        self.nb_images = len(images)
        self.uploader = Uploader()

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
        self.thread = UploadThread(self.images, self.uploadFinished)
        self.thread.upload_complete.connect(self.updateFilename)
        self.thread.upload_complete.connect(self.updateProgressBar)

        # Start the thread
        self.thread.start()


class App(QMainWindow, FileHandling, ErrorHandling):
    def __init__(self):
        QMainWindow.__init__(self)
        FileHandling.__init__(self)
        ErrorHandling.__init__(self)
        self.ui = MainUI()
        self.ui.setupUi(self)

        # connect widgets
        self._connectMenubar()
        self._connectPushButtons()
        self._connectListWidget()

        # set shortcuts
        self.setShortcuts()

        # initally disable pushbuttons
        self.setButtonState(False)

        # accept drag and drops
        self.setAcceptDrops(True)

        self.imgIndex = 0
        self.imgFiles = []
        self.classifications = {}

    def _connectMenubar(self):
        # Connect File actions
        self.ui.actionOpen_Folder.triggered.connect(self.chooseDirDialog)
        self.ui.actionSave.triggered.connect(self.chooseFileDialog)
        self.ui.actionadd_Filter.triggered.connect(self.addFilter)
        self.ui.actionreset_Filters.triggered.connect(self.resetFilters)
        self.ui.actionUpload.triggered.connect(self.uploadImages)

    def _connectPushButtons(self):
        # Connect redo
        self.ui.pushBredo.clicked.connect(self.redoClassification)
        # Connect infected
        self.ui.pushBinf.clicked.connect(lambda: self.classified(True))
        # Connect not infected
        self.ui.pushBnotinf.clicked.connect(lambda: self.classified(False))

    def _connectListWidget(self):
        self.ui.tableWidget.clicked.connect(self.goToListImage)
        self.ui.tableWidget.doubleClicked.connect(self.openListImage)

    def setShortcuts(self):
        """
        Parse and set shortcuts for pushButton trigger events from config file,
        or disable shortcuts.
        """
        # set shortcuts for push buttons
        self.ui.pushBredo.setShortcut("ctrl+z")
        self.ui.pushBnotinf.setShortcut(Qt.Key_Left)
        self.ui.pushBinf.setShortcut(Qt.Key_Right)

    def setButtonState(self, enabled):
        """
        enable or disable all buttons
        """
        self.setInfButtonState(enabled)
        self.setRedoButtonState(enabled)

    def setInfButtonState(self, enabled):
        """
        enable or disable infection buttons
        """
        self.ui.pushBinf.setEnabled(enabled)
        self.ui.pushBnotinf.setEnabled(enabled)

    def setRedoButtonState(self, enabled):
        """
        enable or disable redo button
        """
        self.ui.pushBredo.setEnabled(enabled)

    def chooseFileDialog(self):
        saveDlg = QFileDialog.getSaveFileName(self, "Select file to save results.",
                                              self.dir, "Text files (*.txt *.csv)")
        self.saveFile = saveDlg[0] or None
        if self.saveFile is not None:
            self.saveResults(self.classifications)

    def chooseDirDialog(self):
        # get all image paths
        path = str(QFileDialog.getExistingDirectory(self,
                                                    "Select image directory."))
        if path:
            if len(self.classifications.keys()) != 0:
                msg = QMessageBox()
                msg.setWindowTitle("Warning")
                msg.setText(
                    "Opening a new folder will overwrite previous classifications. Do you want to save your current classifications before opening the new folder?")
                msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
                ans = msg.exec()
                if not ans:
                    return
                self.classifications = {}
                self.imgIndex = 0

            self.dir = path
            self.imgFiles = self.getImagePaths()
            self.initImageDisplay()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()

            if len(self.classifications.keys()) != 0:
                ans = self.askMessageBox(
                    "Warning", "Opening a new folder will overwrite previous classifications. Save current classifications before opening the new folder?")

                if not ans:
                    return

                self.imgIndex = 0
                self.classifications = {}

            # case dir droped
            if os.path.isdir(path):
                self.dir = path
                self.imgFiles = self.getImagePaths()
                self.initImageDisplay()

            # case: image file droped
            elif self.isImageFile(path):
                self.imgFiles = [path]
                self.initImageDisplay()

            elif os.path.isfile(path):
                self.showErrorMessageBox(
                    "Error", "Invalid file type", "Please drag and drop an image file of the following types: TIFF, TIF, JPG, or PNG.")

            else:
                self.showErrorMessageBox(
                    "Error", "File not found"
                )

    def initImageDisplay(self):
        """
        Display first image if one exists
        """
        # load first image
        if len(self.imgFiles) > 0:
            self.displayImage(self.imgFiles[self.imgIndex])
            self.setInfButtonState(True)
            self.loadImgList()
        else:
            self.showErrorMessageBox(
                "Missing input.", "No images files found in directory.",
                "Image files must have one of the following extensions: tif, tiff, png, jpg"
            )

    def displayImage(self, imgpath):
        # load image
        pixmap = QPixmap(imgpath)

        # resize image to fit display area
        DisplaySize = self.ui.display_area.size()
        height, width = DisplaySize.height(), DisplaySize.width()
        pixmap = pixmap.scaled(height, width, Qt.KeepAspectRatio)

        # display image
        self.ui.display_area.setPixmap(pixmap)

    def classified(self, infected):
        """
        save classification and go to next image
        """
        self.classifications[self.imgFiles[self.imgIndex]] = infected
        self.updateImgList(infected)
        self.nextImage(1)

    def redoClassification(self):
        """
        undo classification
        """
        self.nextImage(-1)
        self.updateImgList(False, msg=" ")
        self.classifications.pop(self.imgFiles[self.imgIndex])

    def nextImage(self, move, scrollTop=True):
        self.imgIndex += move

        if scrollTop:
            # scroll to item in list
            self.scrollToCurrent()

        # disable pushButtons if index out of list range
        if self.imgIndex == 0:
            self.setRedoButtonState(False)

        elif self.imgIndex == len(self.imgFiles):
            self.setInfButtonState(False)
        else:
            self.displayImage(self.imgFiles[self.imgIndex])
            self.setButtonState(True)

    def loadImgList(self):
        self.ui.tableWidget.setRowCount(0)
        for row_nb in range(len(self.imgFiles)):
            filename = os.path.splitext(
                os.path.basename(self.imgFiles[row_nb]))[0]

            self.ui.tableWidget.insertRow(row_nb)
            self.ui.tableWidget.setItem(row_nb, 0, QTableWidgetItem(filename))
            self.ui.tableWidget.setItem(row_nb, 1, QTableWidgetItem(""))

    def updateImgList(self, state, msg=""):
        self.ui.tableWidget.setItem(self.imgIndex, 1,
                                    QTableWidgetItem("infected" if state else msg or "not infected"))

    def scrollToCurrent(self):
        item = self.ui.tableWidget.item(self.imgIndex, 0)
        self.ui.tableWidget.scrollToItem(item, QAbstractItemView.PositionAtTop)

    def goToListImage(self, rowItem):
        self.imgIndex = rowItem.row() - 1
        self.nextImage(1, scrollTop=False)

    def openListImage(self, rowItem):
        """
        slot to open image in standard editor by double clicking item on list
        """
        self.openImage(self.imgFiles[rowItem.row()])

    def addFilter(self):
        """
        slot for action "Add Filter" of menubar
        """
        self.filterDlg = FilterDialog(self.imgFiles, self.setFilteredFiles)
        self.filterDlg.show()

    def setFilteredFiles(self, files):
        """
        callback for FilterDialog class to update img file list
        """
        self.imgFiles = files
        self.imgIndex = 0
        self.loadImgList()

    def resetFilters(self):
        self.imgFiles = self.getImagePaths()
        self.loadImgList()

    def uploadImages(self):
        terms = open("terms.txt", "r")
        ans = self.askMessageBox(
            "Terms of Serivce", "Agree to terms of service before upload.", details=terms.read())
        if ans:
            # create uploader object and dialog
            self.uploadDlg = UploadDlg(self.imgFiles)
            self.uploadDlg.exec()


app = QApplication(sys.argv)

# apply dark theme
qdarktheme.setup_theme("auto")

window = App()
window.show()
app.exec()

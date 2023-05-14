"""
GUI to classify infection state of cell cultures.
"""
import sys
import os

from datetime import datetime

import pandas as pd

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import qdarktheme

from filehandling import FileHandling
from errorhandling import ErrorHandling
from dialogs import *

MainUI, QtBaseclass = uic.loadUiType("qt_files/main.ui")


class ConsoleOutputRedirector:
    def __init__(self, textBrowser):
        self.textBrowser = textBrowser

    def write(self, text):
        self.textBrowser.insertPlainText(text)
        self.textBrowser.verticalScrollBar().setValue(
            self.textBrowser.verticalScrollBar().maximum()
        )

    def flush(self):
        pass  # do nothing


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

    def __init__(self, train_data_file, model_save_file, epochs=12, validation_split=0.2, learning_rate=0.001, batch_size=32, rotation=np.pi/4, optimizer="Adam", metrics=["accuracy"]):
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
        self.epochs = epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rotation = rotation
        self.optimizer = optimizer
        self.metrics = metrics


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
        self._connectLineEdit()

        # set shortcuts
        self.setShortcuts()

        # initally disable pushbuttons
        self.setButtonState(False)

        # accept drag and drops
        self.setAcceptDrops(True)

        self.imgIndex = 0
        self.imgFiles = []
        self.classifications = dict()

        self.train_data_file = None
        self.trainParams = None
        self.trainingStarted = False

    def _connectMenubar(self):
        # Connect File actions
        self.ui.actionOpen_Folder.triggered.connect(self.chooseImgDirDlg)
        self.ui.actionSave.triggered.connect(self.chooseResultsFilerDlg)
        self.ui.actionAdd_Filter.triggered.connect(self.addFilter)
        self.ui.actionReset_Filter.triggered.connect(self.resetFilters)
        self.ui.actionUploadImages.triggered.connect(self.uploadImages)
        self.ui.actionUploadModel.triggered.connect(self.uploadModel)
        self.ui.actionUploadClasses.triggered.connect(self.uploadClasses)

    def _reconnectMenubar(self):
        """
        reconnect menubar when tab is switched
        """
        self.ui.actionOpen_Folder.triggered.disconnect()
        self.ui.actionOpen_Folder.triggered.connect(self.chooseTrainDataFile)
        self.ui.actionSave.triggered.disconnect()
        self.ui.actionSave.triggered.connect(self.choseModelSaveFile)

    def _connectPushButtons(self):
        # Connect redo
        self.ui.pushBredo.clicked.connect(self.redoClassification)
        # Connect infected
        self.ui.pushBinf.clicked.connect(lambda: self.classified(True))
        # Connect not infected
        self.ui.pushBnotinf.clicked.connect(lambda: self.classified(False))
        # Connect classify
        self.ui.pushBclassify.clicked.connect(self.automate)

        # Connect push button from second tab
        self.ui.start_train.clicked.connect(self.startTrainModel)
        self.ui.clear.clicked.connect(self.clearTrainLog)

    def _reconnectStartTrainButton(self):
        self.ui.start_train.clicked.disconnect()

        if self.trainingStarted:
            self.ui.start_train.clicked.connect(self.stopTrainModel)
            self.ui.start_train.setText("Stop")
        else:
            self.ui.start_train.clicked.connect(self.startTrainModel)
            self.ui.start_train.setText("Start")

    def _connectListWidget(self):
        self.ui.tableWidget.clicked.connect(self.goToListImage)
        self.ui.tableWidget.doubleClicked.connect(self.openListImage)

    def _connectLineEdit(self):
        self.ui.lineEdit.editingFinished.connect(self.setModelSaveFile)

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

    def chooseResultsFilerDlg(self):
        """
        method to start file dialog for selection of images that are to be classified
        """
        saveDlg = QFileDialog.getSaveFileName(self, "Select file to save results.",
                                              self.dir, "Text files (*.txt *.csv)")
        self.saveFile = saveDlg[0] or None
        if self.saveFile is not None:
            self.saveResults(self.classifications)

    def chooseImgDirDlg(self):
        """
        method to start dir dialog for selection of images that are to be classified
        """
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

    def findChild(self, drop_widget):
        while drop_widget is not None:
            if isinstance(drop_widget, QTableWidget):
                # Found the QTableWidget
                break
            elif isinstance(drop_widget, QLabel):
                break
            elif isinstance(drop_widget, QLineEdit):
                break
            elif isinstance(drop_widget, QAbstractScrollArea):
                # If the widget is a scroll area, get its viewport and check if it contains a QTableWidget
                viewport = drop_widget.viewport()
                if isinstance(viewport, QTableWidget):
                    drop_widget = viewport
                    break
            else:
                drop_widget = drop_widget.parentWidget()
        return drop_widget

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """
        Hanlde a drop event.
        """
        tab = self.ui.tabWidget.currentIndex()

        drop_widget = self.findChild(
            QApplication.instance().widgetAt(QCursor.pos())
        )

        if not tab and drop_widget is self.ui.display_area:
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                self.handleDisplayAreaDrop(path)

        elif tab:
            if drop_widget is self.ui.lineEdit:
                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    self.handleLineEditDrop(path)

            elif drop_widget is self.ui.tableWidget_2:
                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    self.handleTableWidgetDrop(path)

    def handleDisplayAreaDrop(self, path):
        """
        Handle a drop onto the display area.
        """
        if len(self.classifications.keys()) != 0:
            ans = self.askMessageBox(
                "Warning",
                "Opening a new folder will overwrite previous classifications. Save current classifications before opening the new folder?"
            )

            if not ans:
                return

            self.imgIndex = 0
            self.classifications = {}

        if os.path.isdir(path):
            self.dir = path
            self.imgFiles = self.getImagePaths()
            self.initImageDisplay()

        elif self.isImageFile(path):
            self.imgFiles = [path]
            self.initImageDisplay()

        elif os.path.isfile(path):
            self.showErrorMessageBox(
                "Error",
                "Invalid file type",
                "Please drag and drop an image file of the following types: TIFF, TIF, JPG, or PNG."
            )

        else:
            self.showErrorMessageBox("Error", "File not found")

    def handleLineEditDrop(self, path):
        """
        Handle a drop onto the line edit.
        """
        if self.isModelFile(path):
            self.setModelSaveFile(path, update=True)
        else:
            self.showErrorMessageBox(
                "Error",
                "Invalid file type",
                "Please drag and drop an image file of the following type: h5"
            )

    def handleTableWidgetDrop(self, path):
        """
        Handle a drop onto the table widget.
        """
        if self.isTxtFile(path):
            self.setTrainDataFile(path)
        else:
            self.showErrorMessageBox(
                "Warning",
                "Invalid file type",
                "Please drag and drop an image of the following type: .txt, .csv"
            )

    def initImageDisplay(self):
        """
        Display first image if one exists
        """
        # load first image
        if len(self.imgFiles) > 0:
            self.displayImage(self.imgFiles[self.imgIndex])
            self.setInfButtonState(True)
            self.loadImgList(self.ui.tableWidget, self.imgFiles)
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

    #####################################

    def loadImgList(self, tableWidget, imgFiles, labels=None):
        tableWidget.setRowCount(0)
        for row_nb in range(len(imgFiles)):
            filename = os.path.splitext(
                os.path.basename(imgFiles[row_nb]))[0]

            tableWidget.insertRow(row_nb)
            tableWidget.setItem(row_nb, 0, QTableWidgetItem(filename))
            tableWidget.setItem(row_nb, 1, QTableWidgetItem(""))

        if labels is not None:
            for row_nb in range(len(labels)):
                tableWidget.setItem(
                    row_nb, 1, QTableWidgetItem(str(labels[row_nb])))

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
        self.loadImgList(self.ui.tableWidget, self.imgFiles)

    def resetFilters(self):
        self.imgFiles = self.getImagePaths()
        self.loadImgList(self.ui.tableWidget, self.imgFiles)

    #################################

    def uploadImages(self):
        if len(self.imgFiles) == 0:
            self.showErrorMessageBox(
                "Error", "No images to upload.", "Please select a folder with images (via Drag&Drop or menubar)."
            )
            return
        terms = open("terms.txt", "r")
        ans = self.askMessageBox(
            "Terms of Serivce", "Agree to terms of service before upload.", details=terms.read())
        if ans:
            # create uploader object and dialog
            self.uploadDlg = ImgUploadDlg(self.imgFiles)
            self.uploadDlg.exec()
        terms.close()

    def uploadModel(self):
        """
        slot for action "Upload Model" of menubar
        """
        # dialog to get model file (.h5)
        model = QFileDialog.getOpenFileName(
            self, "Select model file.", "h5 files (*.h5)"
        )

        # unpack tuple
        model_arch_path = model[0]

        if model is not None:
            # save model architecture as json
            self.uploadDlg = ModelUploadDlg(model_arch_path)
            self.uploadDlg.exec()

    def uploadClasses(self):
        """
        slot fot action "Upload Classes" of menubar
        """
        self.class_file = QFileDialog.getOpenFileName(
            self, "Select classification file.", "txt files (*.txt *.csv)"
        )[0]

        if os.path.isfile(self.class_file):
            self.uploadDlg = ClassUploadDlg(self.class_file)
            self.uploadDlg.exec()

    ####################################

    def startTrainModel(self):
        """
        slot for push button "Start Training" of second tab
        """
        # get current model save file
        self.setModelSaveFile()

        if not self.isModelFile(self.model_save_file):
            print(self.model_save_file)
            self.showErrorMessageBox(
                "Error",
                "Invalid file type",
                "Please select a model file of the following type: h5"
            )
            return

        if self.train_data_file is not None and self.model_save_file is not None:
            # check if file exists
            if os.path.isfile(self.model_save_file):
                ans = self.askMessageBox(
                    "Warning", "Model file already exists. Overwrite?"
                )
                if not ans:
                    return

            # Redirect the console output to the QPlainTextEdit widget
            sys.stdout = ConsoleOutputRedirector(self.ui.textBrowser)

            # get training parameters
            self.trainParams = self.trainParams or TrainParams(
                self.train_data_file,
                self.model_save_file,
            )

            # train model in seperate thread
            self.train_thread = TrainThread(self.trainParams,
                                            self.ui.textBrowser,
                                            self._reconnectStartTrainButton)
            self.train_thread.start()
            self.trainingStarted = True

        else:
            self.showErrorMessageBox(
                "Error",
                "Missing input",
                "Please select a training data file and a save file."
            )

    def stopTrainModel(self):
        """
        slot for push button "Stop Training" of second tab
        """
        self.train_thread.stop()
        self.ui.start_train.setEnabled(False)
        self.trainingStarted = False

        # wait for thread to finish
        self.train_thread.join()

        # reconnect push button
        self._reconnectStartTrainButton()
        self.ui.clear.setEnabled(True)

    def clearTrainLog(self):
        self.ui.textBrowser.clear()
        self.ui.clear.setEnabled(False)
        self.ui.start_train.setEnabled(True)

    def update_log(self, text):
        self.ui.textBrowser.append(text)

    def setTrainParams(self):
        """ start dialog to set training parameters """
        self.trainParamsDlg = TrainParamsDlg()
        self.trainParamsDlg.exec()
        self.trainParams = self.trainParamsDlg.getParams()

    def setTrainDataFile(self, path):
        """
        Load training data from file and display in table widget.
        """
        self.train_data_file = path

        if self.train_data_file is not None:
            try:
                df = pd.read_csv(self.train_data_file, sep=";", header=0)
                self.loadImgList(self.ui.tableWidget_2,
                                 df["files"].tolist(), df["labels"].tolist())
            # TODO: add more specific error handling (e.g., wrong format, columns, etc.)
            except Exception as e:
                self.showErrorMessageBox("Error", "Invalid file format.",
                                         "File must be a .txt or .csv file with two columns: 'files' and 'labels'.")

    def chooseTrainDataFile(self):
        """
        dialog to choose file in which training data is stored
        """
        saveDlg = QFileDialog.getOpenFileName(self, "Select classifications text file.",
                                              "Text files (*.txt *.csv)")
        self.setTrainDataFile(saveDlg[0] or None)

    def setModelSaveFile(self, filepath=None, update=False):
        """
        slot for lineEdit object to set model save file path
        """
        if filepath is None:
            filepath = self.ui.lineEdit.text()

            if filepath is None:
                return

            dirname = os.path.join(
                os.path.dirname(os.getcwd()),
                os.path.dirname(filepath) or "trained_models"
            )
            filename = os.path.basename(
                filepath) or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

            filepath = os.path.join(dirname, filename)

        self.model_save_file = filepath

        if update:
            self.setLineEditText(filepath)

    def setLineEditText(self, text):
        """
        set text of lineEdit object without triggering signal
        """
        self.ui.lineEdit.blockSignals(True)
        self.ui.lineEdit.setText(text)
        self.ui.lineEdit.blockSignals(False)

    def choseModelSaveFile(self):
        """
        dialog to choose file in which model is stored
        """
        # choosing save file with dialog
        saveDlg = QFileDialog.getSaveFileName(self, "Select file to save results.",
                                              self.dir, "Tensorflow file (*.h5)")
        self.setModelSaveFile(str(saveDlg), update=True)

    ####################################
    def automate(self):
        """
        method to automate classification
        """
        # start AutomateDlg to choose parameters of automated evaluation
        autoDlg = AutomateDlg()
        result = autoDlg.exec()

        if result == QDialog.Accepted:
            # get parameters
            model_path, row_range, col_range, serial_dilution, initial_dilution, particle_to_pfu = autoDlg.getParams()

            # parse params in AutomateConfig object
            automateParams = AutomateConfig(
                self.dir, model_path,
                self.imgFiles, self.classifications,
                row_range, col_range,
                serial_dilution,
                initial_dilution,
                particle_to_pfu,
                self.load_manual_checks  # method to call when classifications is done
            )

            # start classification in seperate thread
            self.classifyThread = ClassifyThread(automateParams)

            self.classifyThread.classify_complete.connect(
                self.update_classify_progress
            )
            self.classifyThread.start()

    def update_classify_progress(self, progress):
        # update progress bar
        self.ui.progressBar.setValue(progress)

    def load_manual_checks(self):
        # load manual checks
        self.classifications = self.classifyThread.classify.get_classifications()
        self.imgList = self.classifications.keys()
        self.imgIndex = 0

        # display first image
        self.loadImgList(self.ui.tableWidget, self.imgList,
                         self.classifications.values())
        self.initImageDisplay()

        # activate push button
        self.setTiterButtonState(True)

    ####################################

    def closeEvent(self, event: QCloseEvent):
        """
        Quit application gracefully.
        """
        if len(self.classifications) > 0:
            reply = QMessageBox.question(
                self, 'Messag', "Unsaved changes. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

        elif self.trainingStarted:
            reply = QMessageBox.question(
                self, 'Messag', "Training in progress. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.stopTrainModel()
                event.accept()
            else:
                event.ignore()

        else:
            event.accept()


app = QApplication(sys.argv)

# apply dark theme
qdarktheme.setup_theme("auto")

window = App()
window.show()
app.exec()

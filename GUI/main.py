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
from params import *
from dialogs import *

MainUI, QtBaseclass = uic.loadUiType("qt_files/main.ui")


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
        self._connectSpinBox()
        self._connectComboBox()
        self._connectCheckBox()

        # set shortcuts
        self.setShortcuts()

        # initally disable pushbuttons
        self.setButtonState(False)

        # accept drag and drops
        self.setAcceptDrops(True)

        self.imgIndex = 0
        self.imgFiles = []
        self.classifications = dict()

        self.trainParams = TrainParams(None, None)

        self.checkForUpdates()

    def _connectMenubar(self):
        # Connect File actions
        self.ui.actionOpen_Folder.triggered.connect(self.chooseImgDirDlg)
        self.ui.actionClassifications.triggered.connect(self.saveClassDlg)
        self.ui.actionTiters.triggered.connect(self.saveTiterDlg)
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
        self.ui.pushBclassify.clicked.connect(self.classify)

        # Connect push button from second tab
        self.ui.start_train.clicked.connect(self.startTrainModel)

    def _connectListWidget(self):
        self.ui.tableWidget.clicked.connect(self.goToListImage)
        self.ui.tableWidget.doubleClicked.connect(self.openListImage)

    def _connectLineEdit(self):
        self.ui.lineEdit.editingFinished.connect(self.setModelSaveFile)

    def _connectSpinBox(self):
        # Connect QSpinBoxes
        self.ui.epochs.valueChanged.connect(self.setEpochs)
        self.ui.batch_size.valueChanged.connect(self.setBatchSize)
        self.ui.learning_rate.valueChanged.connect(self.setLearningRate)
        self.ui.validation_split.valueChanged.connect(self.setValidationSplit)
        self.ui.rotation.valueChanged.connect(self.setRotation)

    def _connectComboBox(self):
        # Connect QComboBoxes
        self.ui.optimizers.currentIndexChanged.connect(self.setOptimizer)

    def _connectCheckBox(self):
        # Connect QCheckBoxes
        self.ui.accuracy.stateChanged.connect(self.setMetrics)
        self.ui.precision.stateChanged.connect(self.setMetrics)
        self.ui.recall.stateChanged.connect(self.setMetrics)

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

    def checkForUpdates(self):
        """
        Check for updates to the application.
        """
        if check_for_updates():
            self.updateDlg = UpdateDlg()
            self.updateDlg.exec()

    def saveClassDlg(self):
        """
        file dialog to choose file to save classifications
        """
        saveDlg = QFileDialog.getSaveFileName(self, "Select file to save classifications.",
                                              self.dir, "Text files (*.txt *.csv)")
        self.saveFile = saveDlg[0] or None
        if self.saveFile is not None:
            self.saveClassifications(self.classifications)

    def saveTiterDlg(self):
        """
        file dialog to choose file to save titers
        """
        if any(classification is None for classification in self.classifications.values()):
            self.showErrorMessageBox(
                "Error",
                "Missing classifications",
                "Please classify all images before saving titers."
            )
            return

        saveDlg = QFileDialog.getSaveFileName(self, "Select file to save titers.",
                                              self.dir, "Text files (*.txt *.csv)")
        self.saveFile = saveDlg[0] or None

        if self.saveFile is not None:
            self.saveTiters(self.get_titers())

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

    def initImageDisplay(self, load_manual_checks=False):
        """
        Display first image if one exists
        """
        # load first image
        if len(self.imgFiles) > 0:
            self.displayImage(self.imgFiles[self.imgIndex])
            self.setInfButtonState(True)

            # TODO: very ugly code, refactor
            # this is to prevent the table from being loaded twice
            # when loading manual checks
            if load_manual_checks:
                self.loadImgList(self.ui.tableWidget, self.imgFiles,
                                 list(self.classifications.values()))
            else:
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

        # display next image
        self.displayImage(self.imgFiles[self.imgIndex])

        # disable redo button if no classifications have been made
        if self.imgFiles[self.imgIndex] not in self.classifications.keys():
            self.setRedoButtonState(False)
        else:
            self.setRedoButtonState(True)

        # disable classify buttons if index out of list range
        if self.imgIndex == 0:
            self.setRedoButtonState(False)

        elif self.imgIndex == len(self.imgFiles):
            self.setInfButtonState(False)
        else:
            self.setInfButtonState(True)

    #####################################

    def loadImgList(self, tableWidget, imgFiles, labels=None):
        tableWidget.setRowCount(0)

        if labels is None:
            labels = [None] * len(imgFiles)

        for row_nb in range(len(imgFiles)):
            filename = os.path.splitext(
                os.path.basename(imgFiles[row_nb])
            )[0]

            if labels[row_nb] is not None:
                label = "infected" if labels[row_nb] == 1 else "not infected"
            else:
                label = ""

            tableWidget.insertRow(row_nb)

            # set filename
            tableWidget.setItem(row_nb, 0, QTableWidgetItem(filename))
            # set infection label
            tableWidget.setItem(row_nb, 1, QTableWidgetItem(label))

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
        print(rowItem.row(), self.imgFiles[rowItem.row()])
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
        self.initImageDisplay()

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

        if not self.isModelFile(self.trainParams.model_save_file):
            self.showErrorMessageBox(
                "Error",
                "Invalid file type",
                "Please select a model file of the following type: h5"
            )
            return

        if self.trainParams.train_data_file is not None and self.trainParams.model_save_file is not None:
            # check if file exists
            if os.path.isfile(self.trainParams.model_save_file):
                improve = self.askImproveMessageBox(
                    "Warning", "Model file already exists. Do you want to improve upon the existing model?"
                )
                if improve is None:
                    return

            self.trainParams.train_mode = "transfer" if improve else "replace"

            # train model in seperate thread
            self.train_thread = TrainThread(self.trainParams)
            self.train_thread.start()

        else:
            self.showErrorMessageBox(
                "Error",
                "Missing input",
                "Please select a training data file and a save file."
            )

    def setEpochs(self):
        self.trainParams.epochs = int(self.ui.epochs.value())

    def setBatchSize(self):
        self.trainParams.batch_size = int(self.ui.batch_size.value())

    def setLearningRate(self):
        self.trainParams.learning_rate = float(self.ui.learning_rate.value())

    def setOptimizer(self):
        self.trainParams.optimizer = self.ui.optimizers.currentText()

    def setValidationSplit(self):
        self.trainParams.validation_split = float(
            self.ui.validation_split.value())

    def setDropout(self):
        self.trainParams.dropout = float(self.ui.dropout.value())

    def setRotation(self):
        self.trainParams.rotation = int(self.ui.rotation.value())

    def setHorizontalFlip(self):
        self.trainParams.horiz_flip = self.ui.horizontal_flip.isChecked()

    def setVerticalFlip(self):
        self.trainParams.vert_flip = self.ui.vertical_flip.isChecked()

    def setMetrics(self):
        self.trainParams.metrics = []
        if self.ui.accuracy.isChecked():
            self.trainParams.metrics.append("accuracy")
        if self.ui.precision.isChecked():
            self.trainParams.metrics.append("precision")
        if self.ui.recall.isChecked():
            self.trainParams.metrics.append("recall")

    def setTrainDataFile(self, path):
        """
        Load training data from file and display in table widget.
        """
        self.trainParams.train_data_file = path

        if self.trainParams.train_data_file is not None:
            try:
                df = pd.read_csv(self.trainParams.train_data_file,
                                 sep=";", header=0)
                self.loadImgList(self.ui.tableWidget_2,
                                 df["images"].tolist(), df["labels"].tolist())
            # TODO: add more specific error handling (e.g., wrong format, columns, etc.)
            except Exception as e:
                self.showErrorMessageBox("Error", "Invalid file format.",
                                         "File must be a .txt or .csv file with two columns: 'images' and 'labels'.")

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

        self.trainParams.model_save_file = filepath

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
    def classify(self):
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
                self.sort_manual_checks  # method to call when classifications is done
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

    def sort_manual_checks(self):
        """
        This methode is called when classification is done.
        It sorts images such that non-classified images are displayed first.
        """
        # get sorted list of images
        self.classifications = self.classifyThread.classify.get_unclassified_images()
        self.imgFiles = list(self.classifications.keys())
        self.imgIndex = 0

        # reload list
        self.loadImgList(self.ui.tableWidget, self.imgFiles,
                         list(self.classifications.values()))

        # display first image
        self.initImageDisplay(load_manual_checks=True)

    def get_titers(self):
        """"""
        manual_checks = self.classifyThread.classify.get_manual_checks()

        manually_checked = dict()

        # add manual checks to classifications
        for image in manual_checks.keys():
            row, col = manual_checks[image]
            label = self.classifications[image]

            manually_checked[image] = (row, col, label)

        self.classifyThread.classify.set_manual_checks(manually_checked)
        self.titers = self.classifyThread.classify.get_titers()

        return self.titers

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

        else:
            event.accept()


app = QApplication(sys.argv)

# apply dark theme
qdarktheme.setup_theme("auto")

window = App()
window.show()
app.exec()

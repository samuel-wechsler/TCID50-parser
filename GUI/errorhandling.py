from PyQt5.QtWidgets import *


class ErrorHandling(object):
    """
    Class for displaying errors to the user
    """

    def __init__(self):
        self.preventClose = False

    def closeEvent(self, event):
        """
        This is the event handler for the QDialog.closeEvent.
        """
        # Normally the editingFinished signal is only called after this event is handled.
        # However we need to call it now in order to ignore the close event in case of an error.
        widget = self.focusWidget()
        if widget is not None:
            try:
                widget.editingFinished.emit()
                # block the signals to prevent triggering the editingFinished signal twice
                # if there is an error the signals will get enabled again when the errorMessageBox is closed
                widget.blockSignals(True)
            except AttributeError:
                pass

        # Prevent the dialog from closing if there is an error
        if self.preventClose:
            event.ignore()

    def showErrorMessageBoxInEditingFinished(self, widget, title, message, details="", preventClose=True):
        """
        Displays a QMessageBox with the given errorMsg and takes a widget in order to prevent the editingFinished signal on this widget to be emitted twice.
        This would happen because of the following bug:
        https://stackoverflow.com/questions/37458898/editingfinished-of-qlineedit-is-emited-a-second-time-when-a-dialog-is-executed-i
        """
        self.preventClose = preventClose
        widget.blockSignals(True)

        def callback(_event):
            self.preventClose = False
            widget.blockSignals(False)

        self.msgBox = QMessageBox()
        self.msgBox.closeEvent = callback
        self.msgBox.buttonClicked.connect(callback)
        self.msgBox.setWindowTitle(title)
        self.msgBox.setText(message)
        self.msgBox.setDetailedText(details)
        self.msgBox.show()

    def showErrorMessageBox(self, title, message, details=""):
        """
        Displays a QMessageBox with the given errorMsg.
        """
        self.msgBox = QMessageBox()
        self.msgBox.setWindowTitle(title)
        self.msgBox.setText(message)
        self.msgBox.setDetailedText(details)
        self.msgBox.show()

    def askMessageBox(self, title, message, details=""):
        self.msg = QMessageBox()
        self.msg.setWindowTitle(title)
        self.msg.setText(message)
        self.msg.setDetailedText(details)
        self.msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        ans = self.msg.exec()
        if ans == QMessageBox.Cancel:
            return False
        return True

    def askImproveMessageBox(self, title, message, details=""):
        self.msg = QMessageBox()
        self.msg.setWindowTitle(title)
        self.msg.setText(message)
        self.msg.setDetailedText(details)
        self.msg.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        ans = self.msg.exec()
        if ans == QMessageBox.Cancel:
            return None
        elif ans == QMessageBox.Yes:
            return True
        elif ans == QMessageBox.No:
            return False

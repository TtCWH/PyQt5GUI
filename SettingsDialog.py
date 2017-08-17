from PyQt5 import QtCore, QtGui, QtWidgets
from SettingsDialogUI import Ui_Dialog

class SettingsDialog(Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__()
        self.setupUi(self)

    def handle_click(self):
        if not self.isVisible():
            self.show()


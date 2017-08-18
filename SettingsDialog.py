from PyQt5 import QtCore, QtGui, QtWidgets
from SettingsDialogUI import Ui_Dialog
import Parameters

class SettingsDialog(Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__()
        self.setupUi(self)
        self.buttonBox.rejected.connect(self.reject)
        self.pushButton_2.clicked.connect(lambda: self.apply_settings(False))
        self.pushButton.clicked.connect(self.reset_settings)
        self.buttonBox.accepted.connect(lambda: self.apply_settings(True))

    def handle_click(self):
        if not self.isVisible():
            self.show()
            self.lineEdit.setText(str(Parameters.ValidationPercentage))
            self.lineEdit_2.setText(str(Parameters.TestSetPercentage))
            self.lineEdit_3.setText(str(Parameters.LearningRate))
            self.lineEdit_4.setText(str(Parameters.LearningSteps))
            self.lineEdit_5.setText(str(Parameters.BatchSize))

    def apply_settings(self, tag):
        Parameters.ValidationPercentage = int(self.lineEdit.text())
        Parameters.TestSetPercentage = int(self.lineEdit_2.text())
        Parameters.LearningRate = float(self.lineEdit_3.text())
        Parameters.LearningSteps = int(self.lineEdit_4.text())
        Parameters.BatchSize = int(self.lineEdit_5.text())
        if tag and self.isVisible():
            self.hide()

    def reset_settings(self):
        self.lineEdit.setText(str(10))
        self.lineEdit_2.setText(str(20))
        self.lineEdit_3.setText(str(0.01))
        self.lineEdit_4.setText(str(5000))
        self.lineEdit_5.setText(str(32))





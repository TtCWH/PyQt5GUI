from PyQt5 import QtCore, QtGui, QtWidgets
from SettingsDialogUI import Ui_Dialog
import Parameters
import sys


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

    def adjust_parameters(self, to_be_adjusted, low_bound, high_bound):
        if to_be_adjusted < low_bound:
            to_be_adjusted = low_bound
        elif to_be_adjusted > high_bound:
            to_be_adjusted = high_bound
        return to_be_adjusted

    def apply_settings(self, tag):
        try:
            validation_percentage = self.adjust_parameters(int(self.lineEdit.text()), 0, 50)
            testset_percentage = self.adjust_parameters(int(self.lineEdit_2.text()), 0, 50)
            learning_rate = self.adjust_parameters(float(self.lineEdit_3.text()), 0.0, float("inf"))
            learning_steps = self.adjust_parameters(float(self.lineEdit_4.text()), 0, sys.maxsize)
            batch_size = self.adjust_parameters(int(self.lineEdit_5.text()), 0, sys.maxsize)

            Parameters.ValidationPercentage = validation_percentage
            Parameters.TestSetPercentage = testset_percentage
            Parameters.LearningRate = learning_rate
            Parameters.LearningSteps = learning_steps
            Parameters.BatchSize = batch_size
            if tag and self.isVisible():
                self.hide()
        except:
            self.hide()
            warning = QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Your input is invalid, please input again.')
            self.handle_click()

    def reset_settings(self):
        self.lineEdit.setText(str(10))
        self.lineEdit_2.setText(str(20))
        self.lineEdit_3.setText(str(0.01))
        self.lineEdit_4.setText(str(5000))
        self.lineEdit_5.setText(str(32))

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import Parameters
import MainGUI
import SettingsDialog

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainGUI = MainGUI.MainGUI()
    settings = SettingsDialog.SettingsDialog()
    mainGUI.SettingsButton.clicked.connect(settings.handle_click)
    settings.buttonBox.accepted.connect(mainGUI.learning_status_init)

    sys.exit(app.exec_())


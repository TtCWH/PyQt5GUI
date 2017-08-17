import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import MainGUI
import SettingsDialog

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainGUI.MainGUI()
    settings = SettingsDialog.SettingsDialog()
    main.pushButton.clicked.connect(settings.handle_click)
    sys.exit(app.exec_())
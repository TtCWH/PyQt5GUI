import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import Parameters
import MainGUI
import SettingsDialog

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainGUI.MainGUI()
    settings = SettingsDialog.SettingsDialog()
    main.pushButton.clicked.connect(settings.handle_click)
    settings.buttonBox.accepted.connect(main.learning_status_init)
    sys.exit(app.exec_())
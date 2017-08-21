import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import Parameters
import MainGUI
import SettingsDialog
import json

if __name__ == '__main__':
    config_file = open('Config/config.json', 'r')
    config_dic = json.load(config_file)
    config_file.close()
    Config = Parameters.Parameters(config_dic)
    Config.adjust_parameters()
    #print(Parameters.N_CLASSES)
    #print(Config.config)

    app = QtWidgets.QApplication(sys.argv)
    mainGUI = MainGUI.MainGUI()
    settings = SettingsDialog.SettingsDialog()
    mainGUI.SettingsButton.clicked.connect(settings.handle_click)
    settings.buttonBox.accepted.connect(mainGUI.learning_status_init)

    Config.save_configs()

    sys.exit(app.exec_())


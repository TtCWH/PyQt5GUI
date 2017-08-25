from ReadmeUI import Ui_Dialog
from PyQt5 import QtCore, QtGui, QtWidgets

class Readme(Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, text_data="", parent=None):
        super(Readme, self).__init__()
        self.setupUi(self)
        self.textBrowser.setText(text_data)
        self.show()

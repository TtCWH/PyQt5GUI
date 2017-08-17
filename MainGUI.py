from PyQt5 import QtCore, QtGui, QtWidgets
from MainWindowUI import Ui_MainWindow

class MainGUI(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainGUI, self).__init__()
        self.setupUi(self)
        self.show()

        self.pushButton_2.clicked.connect(lambda: self.ShowFileDialog(2))
        self.pushButton_7.clicked.connect(lambda: self.ShowFileDialog(7))

    def ShowFileDialog(self, n):
        if n == 2:
            fname = QtWidgets.QFileDialog.getExistingDirectory()
            self.lineEdit.setText(fname)
        elif n == 7:
            fname = QtWidgets.QFileDialog.getOpenFileName()
            self.lineEdit_2.setText(fname[0])



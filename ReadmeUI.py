# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Readme.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(520, 460)
        Dialog.setMinimumSize(QtCore.QSize(520, 460))
        Dialog.setMaximumSize(QtCore.QSize(520, 460))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Icons/Readme.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 520, 460))
        self.textBrowser.setMinimumSize(QtCore.QSize(520, 460))
        self.textBrowser.setMaximumSize(QtCore.QSize(520, 460))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "ReadMe"))


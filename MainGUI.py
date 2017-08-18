from tensorflow.python.platform import gfile
from PyQt5 import QtCore, QtGui, QtWidgets
from MainWindowUI import Ui_MainWindow
import Parameters
import MigrateTraining
import tensorflow as tf
import os.path
import os
import shutil
import WorkThread


class MainGUI(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainGUI, self).__init__()
        self.setupUi(self)
        self.show()

        self.pushButton_2.clicked.connect(lambda: self.ShowFileDialog(2))
        self.pushButton_7.clicked.connect(lambda: self.ShowFileDialog(7))
        self.pushButton_3.clicked.connect(self.start_training)
        self.pushButton_3.clicked.connect(lambda: self.update_textedit("Start training...\n"))
        self.pushButton_4.clicked.connect(self.save_results)
        self.pushButton_5.clicked.connect(self.delete_model)
        self.pushButton_8.clicked.connect(self.picture_predict)

    def delete_model(self):
        reply = QtWidgets.QMessageBox.warning(self, 'Delete Model', 'Delete the saved model?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            shutil.rmtree(r'Models/')
            os.mkdir(r'Models')
            self.textEdit.setText("")
            QtWidgets.QMessageBox.information(self, 'Delete Done', 'You have deleted the model.')
            self.progressBar.setValue(0)


    def ShowFileDialog(self, n):
        if n == 2:
            fname= QtWidgets.QFileDialog.getExistingDirectory()
            if fname:
                self.lineEdit.setText(fname)
                Parameters.INPUT_DATA = fname
        elif n == 7:
            fname = QtWidgets.QFileDialog.getOpenFileName()
            if fname[0]:
                self.lineEdit_2.setText(fname[0])
                self.textEdit_2.setText("")

    def learning_status_init(self):
        status_info = "Validation Percentage: %d\nTestSet Percentage: %d\n" \
                      "Learning Rate: %f\nLearning Steps: %d\nBatch Size: %d\n" \
                      % (Parameters.ValidationPercentage, Parameters.TestSetPercentage, Parameters.LearningRate,
                         Parameters.LearningSteps, Parameters.BatchSize)
        self.textEdit.setText(status_info)

    def check_status(self):
        pass

    def update_textedit(self, status_info):
        self.textEdit.append(status_info)

    def update_predict(self, res):
        self.textEdit_2.append("This image belongs to class %d.\n" %(res))

    def update_processBar(self, percentage):
        self.progressBar.setValue(percentage)

    def save_results(self):
        res_info = self.textEdit.toPlainText()
        fname= QtWidgets.QFileDialog.getSaveFileName(directory='Results/')
        if fname[0]:
            res_save = open(fname[0], 'w')
            res_save.write(res_info)
            res_save.close()
            res_save.close()

    def start_training(self):
        self.TrainThread = WorkThread.TraingProcess()
        self.TrainThread.status_info.connect(self.update_textedit)
        self.TrainThread.done_percentage.connect(self.update_processBar)
        self.TrainThread.start()

    def picture_predict(self):
        self.textEdit_2.setText("Start predicting, please wait a moment...\n")
        try:
            image_Path = self.lineEdit_2.text()
            self.PredictThread = WorkThread.PicturePredict(image_Path)
            self.PredictThread.trigger.connect(self.update_predict)
            self.PredictThread.start()
        except:
            warning = QtWidgets.QMessageBox.warning(self, 'Invalied Image',
                                                    'Invalid Image:\n What you selected was not an image or the filename contains invalid characters.')














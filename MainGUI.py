from tensorflow.python.platform import gfile
from PyQt5 import QtCore, QtGui, QtWidgets
from MainWindowUI import Ui_MainWindow
import Parameters
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
            try:
                os.mkdir(r'Models')
                self.textEdit.setText("")
                QtWidgets.QMessageBox.information(self, 'Delete Done', 'You have deleted the model.')
                self.progressBar.setValue(0)
            except:
                QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


    def ShowFileDialog(self, n):
        if n == 2:
            fname= QtWidgets.QFileDialog.getExistingDirectory()
            if fname:
                self.lineEdit.setText(fname)
                Parameters.INPUT_DATA = fname
                #print(Parameters.INPUT_DATA)
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

    def update_textedit(self, status_info):
        old_info = self.textEdit.toPlainText()
        new_info = status_info + '\n'+ old_info
        self.textEdit.setText(new_info)



    def update_predict(self, res):
        if res < 0:
            self.textEdit_2.setText("")
            QtWidgets.QMessageBox.warning(self, 'Error', 'No model found!\nPlease train a model first.')
        else:
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

    def disable_buttons(self):
        self.pushButton_2.setDisabled(True)
        self.pushButton.setDisabled(True)
        self.pushButton_3.setDisabled(True)
        self.pushButton_4.setDisabled(True)
        self.pushButton_5.setDisabled(True)
        self.pushButton_7.setDisabled(True)
        self.pushButton_8.setDisabled(True)

    def enable_buttons(self, tag):
        self.pushButton_2.setEnabled(tag)
        self.pushButton.setEnabled(tag)
        self.pushButton_3.setEnabled(tag)
        self.pushButton_4.setEnabled(tag)
        self.pushButton_5.setEnabled(tag)
        self.pushButton_7.setEnabled(tag)
        self.pushButton_8.setEnabled(tag)

    def recovery_gui(self, tag):
        if tag == 1:
            self.textEdit.setText("")
            self.lineEdit.setText("")
        if tag == 2:
            self.textEdit_2.setText("")
            self.lineEdit_2.setText("")
        self.progressBar.setValue(0)
        self.enable_buttons(True)

    def error_process(self, error_key):
        if error_key == 1:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not load image list, please check your image set.\n")
            self.recovery_gui(1)
        if error_key == 2:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not get the image data, please check your input.\n")
            self.recovery_gui(2)
        if error_key == 3:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not form the train bottlenecks, please check your image set.\n")
            self.recovery_gui(1)
        if error_key == 4:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not form the validation bottlenecks, please check your image set.\n")
            self.recovery_gui(1)
        if error_key == 5:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not form the test bottlenecks, please check your image set.\n")
            self.recovery_gui(1)

    def start_training(self):
        self.disable_buttons()
        self.TrainThread = WorkThread.TrainingProcess()
        self.TrainThread.status_info.connect(self.update_textedit)
        self.TrainThread.done_percentage.connect(self.update_processBar)
        self.TrainThread.finish_signal.connect(self.enable_buttons)
        self.TrainThread.error_signal.connect(self.error_process)
        self.TrainThread.start()

    def picture_predict(self):
        self.disable_buttons()
        self.textEdit_2.setText("Start predicting, please wait a moment...\n")
        try:
            image_Path = self.lineEdit_2.text()
            self.PredictThread = WorkThread.PicturePredict(image_Path)
            self.PredictThread.trigger.connect(self.update_predict)
            self.PredictThread.finish_signal.connect(self.enable_buttons)
            self.PredictThread.error_signal.connect(self.error_process)
            self.PredictThread.start()
        except:
            QtWidgets.QMessageBox.warning(self, 'Invalied Image', 'Invalid Image.')














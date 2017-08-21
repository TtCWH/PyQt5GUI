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

        self.ImageSetBrowseButton.clicked.connect(lambda: self.ShowFileDialog(1))
        self.ImageBrowseButton.clicked.connect(lambda: self.ShowFileDialog(2))
        self.StartTrainingButton.clicked.connect(self.start_training)
        self.StartTrainingButton.clicked.connect(lambda: self.update_statusinfo("Start training...\n"))
        self.SaveResButton.clicked.connect(self.save_results)
        self.DelModelButton.clicked.connect(self.delete_model)
        self.StartPredictButton.clicked.connect(self.picture_predict)

    def delete_model(self):
        reply = QtWidgets.QMessageBox.warning(self, 'Delete Model', 'Delete the saved model?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                if os.path.exists('Models'):
                    shutil.rmtree(r'Models/')
                os.mkdir(r'Models')
                self.TrainingStatusBrowser.setText("")
                QtWidgets.QMessageBox.information(self, 'Delete Done', 'You have deleted the model.')
                self.progressBar.setValue(0)
            except:
                QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


    def ShowFileDialog(self, tag):
        if tag == 1:
            try:
                fname = QtWidgets.QFileDialog.getExistingDirectory()
                if fname:
                    self.ImageSetPath.setText(fname)
                    Parameters.INPUT_DATA = fname
                    #print(Parameters.INPUT_DATA)
            except:
                QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')
        elif tag == 2:
            try:
                fname = QtWidgets.QFileDialog.getOpenFileName()
                if fname[0]:
                    self.ImagePath.setText(fname[0])
                    self.PredictResBrowser.setText("")
            except:
                QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')

    def learning_status_init(self):
        status_info = "Validation Percentage: %d\nTestSet Percentage: %d\n" \
                      "Learning Rate: %f\nLearning Steps: %d\nBatch Size: %d\n" \
                      % (Parameters.ValidationPercentage, Parameters.TestSetPercentage, Parameters.LearningRate,
                         Parameters.LearningSteps, Parameters.BatchSize)
        self.TrainingStatusBrowser.setText(status_info)

    def update_statusinfo(self, status_info):
        self.TrainingStatusBrowser.append(status_info)
        self.TrainingStatusBrowser.moveCursor(QtGui.QTextCursor.End)
        #old_info = self.TrainingStatusBrowser.toPlainText()
        #new_info = status_info + '\n'+ old_info
        #self.TrainingStatusBrowser.setText(new_info)


    def update_predict(self, res):
        if res < 0:
            self.PredictResBrowser.setText("")
            QtWidgets.QMessageBox.warning(self, 'Error', 'No model found!\nPlease train a model first.')
        else:
            self.PredictResBrowser.append("This image belongs to '%s'.\n" %(Parameters.LABEL_NAME_LIST[res]))

    def update_processBar(self, percentage):
        self.progressBar.setValue(percentage)

    def save_results(self):
        res_info = self.TrainingStatusBrowser.toPlainText()
        try:
            fname= QtWidgets.QFileDialog.getSaveFileName(directory='Results/')
            if fname[0]:
                res_save = open(fname[0], 'w')
                res_save.write(res_info)
                res_save.close()
        except:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


    def disable_buttons(self):
        self.ImageSetBrowseButton.setDisabled(True)
        self.SettingsButton.setDisabled(True)
        self.StartTrainingButton.setDisabled(True)
        self.DelModelButton.setDisabled(True)
        self.SaveResButton.setDisabled(True)
        self.ImageBrowseButton.setDisabled(True)
        self.StartPredictButton.setDisabled(True)
        self.WrongButton.setDisabled(True)
        self.RightButton.setDisabled(True)

    def enable_buttons(self, tag):
        self.ImageSetBrowseButton.setEnabled(tag)
        self.SettingsButton.setEnabled(tag)
        self.StartTrainingButton.setEnabled(tag)
        self.DelModelButton.setEnabled(tag)
        self.SaveResButton.setEnabled(tag)
        self.ImageBrowseButton.setEnabled(tag)
        self.StartPredictButton.setEnabled(tag)
        self.WrongButton.setEnabled(tag)
        self.RightButton.setEnabled(tag)

    def recovery_gui(self, tag):
        if tag == 1:
            self.TrainingStatusBrowser.setText("")
            self.ImageSetPath.setText("")
        if tag == 2:
            self.PredictResBrowser.setText("")
            self.ImagePath.setText("")
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
        self.TrainThread.status_info.connect(self.update_statusinfo)
        self.TrainThread.done_percentage.connect(self.update_processBar)
        self.TrainThread.finish_signal.connect(self.enable_buttons)
        self.TrainThread.error_signal.connect(self.error_process)
        self.TrainThread.start()

    def picture_predict(self):
        self.disable_buttons()
        self.PredictResBrowser.setText("Start predicting, please wait a moment...\n")
        try:
            image_Path = self.ImagePath.text()
            self.PredictThread = WorkThread.PicturePredict(image_Path)
            self.PredictThread.trigger.connect(self.update_predict)
            self.PredictThread.finish_signal.connect(self.enable_buttons)
            self.PredictThread.error_signal.connect(self.error_process)
            self.PredictThread.start()
        except:
            QtWidgets.QMessageBox.warning(self, 'Invalied Image', 'Invalid Image.\nPlease check your input.')














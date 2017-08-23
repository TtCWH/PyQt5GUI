from PyQt5 import QtCore, QtGui, QtWidgets
from MainWindowUI import Ui_MainWindow
import Parameters
import os.path
import os
import shutil
import WorkThread
import json
import ChooseRightLabel


class MainGUI(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainGUI, self).__init__()
        self.setupUi(self)

        self.disable_buttons()
        self.Config = Parameters.Parameters({})
        self.show()

        self.LoadConfigButton.clicked.connect(self.load_configures)
        self.ImageSetBrowseButton.clicked.connect(lambda: self.ShowFileDialog(1))
        self.ImageBrowseButton.clicked.connect(lambda: self.ShowFileDialog(2))
        self.StartTrainingButton.clicked.connect(self.start_training)
        self.StartTrainingButton.clicked.connect(lambda: self.update_statusinfo("Start training...\n"))
        self.SaveResButton.clicked.connect(self.save_results)
        self.DelModelButton.clicked.connect(self.delete_model)
        self.StartPredictButton.clicked.connect(self.picture_predict)
        self.WrongButton.clicked.connect(lambda: self.wrong_prediction(self.feed_backs))
        self.RightButton.clicked.connect(lambda: self.right_prediction(self.feed_backs))


    def closeEvent(self, *args, **kwargs):
        # 关闭窗口时保存配置
        try:
            reply = QtWidgets.QMessageBox.question(self, 'Save configures', 'Please save your own configures.',
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save configure file', directory='Config',
                                                              filter='configs(*.json)')
                if fname[0]:
                    self.Config.save_configs(fname[0])
        except:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


    def load_configures(self):
        try:
            # 载入配置文件
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Load configure file', directory='Config')
            if fname[0]:
                config_file = open(fname[0], 'r')
                config_dic = json.load(config_file)
                config_file.close()
                self.Config = Parameters.Parameters(config_dic)
                self.Config.adjust_parameters()
                QtWidgets.QMessageBox.information(self, 'Success', 'You have loaded your configure file!')
                self.enable_buttons(1)
        except:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


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
                fname = QtWidgets.QFileDialog.getExistingDirectory(directory='datasets')
                if fname:
                    self.ImageSetPath.setText(fname)
                    Parameters.INPUT_DATA = fname
                    # print(Parameters.INPUT_DATA)
            except:
                QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')
        elif tag == 2:
            try:
                fname = QtWidgets.QFileDialog.getOpenFileName(directory='datasets')
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
        # old_info = self.TrainingStatusBrowser.toPlainText()
        # new_info = status_info + '\n'+ old_info
        # self.TrainingStatusBrowser.setText(new_info)


    def update_predict(self, res):
        if res < 0:
            self.PredictResBrowser.setText("")
            QtWidgets.QMessageBox.warning(self, 'Error', 'No model found!\nPlease train a model first.')
        else:
            self.PredictResBrowser.append("This image belongs to '%s'.\n" % (Parameters.LABEL_NAME_LIST[res]))


    def update_processBar(self, percentage):
        self.progressBar.setValue(percentage)


    def save_results(self):
        res_info = self.TrainingStatusBrowser.toPlainText()
        try:
            fname = QtWidgets.QFileDialog.getSaveFileName(directory='Results/')
            if fname[0]:
                res_save = open(fname[0], 'w')
                res_save.write(res_info)
                res_save.close()
        except:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


    def disable_buttons(self, tag=1):
        if tag == 2:
            self.LoadConfigButton.setDisabled(True)
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
        #self.detect_thread_status()
        self.ImageSetBrowseButton.setEnabled(True)
        self.SettingsButton.setEnabled(True)
        self.StartTrainingButton.setEnabled(True)
        self.DelModelButton.setEnabled(True)
        self.SaveResButton.setEnabled(True)
        self.ImageBrowseButton.setEnabled(True)
        self.StartPredictButton.setEnabled(True)
        self.LoadConfigButton.setEnabled(True)

        if tag == 2:
            self.WrongButton.setEnabled(True)
            self.RightButton.setEnabled(True)


        #self.restart_threads()


    def recovery_gui(self, tag):
        if tag == 1:
            self.TrainingStatusBrowser.setText("")
            self.ImageSetPath.setText("")
        if tag == 2:
            self.PredictResBrowser.setText("")
            self.ImagePath.setText("")
        self.progressBar.setValue(0)
        self.enable_buttons(1)


    def error_process(self, error_key):
        if error_key == 1:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not load image list, please check your image set.\n")
            self.recovery_gui(1)
        if error_key == 2:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not get the image data, please check your input.\n")
            self.recovery_gui(2)
        if error_key == 3:
            QtWidgets.QMessageBox.warning(self, 'Error',
                                          "Can not form the train bottlenecks, please check your image set.\n")
            self.recovery_gui(1)
        if error_key == 4:
            QtWidgets.QMessageBox.warning(self, 'Error',
                                          "Can not form the validation bottlenecks, please check your image set.\n")
            self.recovery_gui(1)
        if error_key == 5:
            QtWidgets.QMessageBox.warning(self, 'Error',
                                          "Can not form the test bottlenecks, please check your image set.\n")
            self.recovery_gui(1)
        if error_key == 6:
            QtWidgets.QMessageBox.warning(self, 'Error', "Can not get the image bottleneck, please check your input.\n")
            self.recovery_gui(1)


    def wrong_prediction(self, feed_backs):
        self.WrongButton.setDisabled(True)
        self.RightButton.setDisabled(True)
        try:
            reply = QtWidgets.QMessageBox.question(self, 'Oops',
                                               'Oops...Sorry for a bad prediction.\nDo you want to make a feedback?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.ChooseDialog = ChooseRightLabel.ChooseRightLabel(feed_backs[1], feed_backs[2])
        except:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


    def right_prediction(self, feed_backs):
        self.WrongButton.setDisabled(True)
        self.RightButton.setDisabled(True)
        try:
            label = Parameters.LABEL_NAME_LIST[feed_backs[0]]
            dest_path = os.path.join(Parameters.TRAININGDATABASE, Parameters.MODEL_SAVE_NAME, label)
            shutil.copy(feed_backs[1], dest_path)
            QtWidgets.QMessageBox.information(self, 'Nice', 'A nice prediction. So happy!')
            if os.path.exists(feed_backs[1]):
                os.remove(feed_backs[1])
        except:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Something was wrong!')


    def handle_feed_back(self, feed_back_list):
        self.PredictThread.stop()
        while not self.PredictThread.isFinished():
            self.PredictThread.wait()
        self.feed_backs = feed_back_list


    def start_training(self):
        self.disable_buttons(tag=2)
        self.TrainThread = WorkThread.TrainingProcess()
        self.TrainThread.status_info.connect(self.update_statusinfo)
        self.TrainThread.done_percentage.connect(self.update_processBar)
        self.TrainThread.finish_signal.connect(lambda: self.enable_buttons(1))
        self.TrainThread.error_signal.connect(self.error_process)
        self.TrainThread.start()

    def picture_predict(self):
        self.disable_buttons(tag=2)
        self.PredictResBrowser.setText("Start predicting, please wait a moment...\n")
        Parameters.input_image_path = self.ImagePath.text()
        try:
            self.PredictThread = WorkThread.PicturePredict()
            self.PredictThread.trigger.connect(self.update_predict)
            self.PredictThread.finish_signal.connect(lambda: self.enable_buttons(2))
            self.PredictThread.error_signal.connect(self.error_process)
            self.PredictThread.feed_back.connect(self.handle_feed_back)
            self.PredictThread.start()
        except:
            QtWidgets.QMessageBox.warning(self, 'Invalied Image', 'Invalid Image.\nPlease check your input.')

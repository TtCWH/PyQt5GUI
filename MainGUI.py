from tensorflow.python.platform import gfile
from PyQt5 import QtCore, QtGui, QtWidgets
from MainWindowUI import Ui_MainWindow
import Parameters
import MigrateTraining
import tensorflow as tf
import os.path
import os
import shutil



class MainGUI(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainGUI, self).__init__()
        self.setupUi(self)
        self.show()

        self.pushButton_2.clicked.connect(lambda: self.ShowFileDialog(2))
        self.pushButton_7.clicked.connect(lambda: self.ShowFileDialog(7))
        self.pushButton_3.clicked.connect(lambda: self.update_textedit("Start training...\n"))
        self.pushButton_3.clicked.connect(self.start_training)
        self.pushButton_4.clicked.connect(self.save_results)
        self.pushButton_5.clicked.connect(self.delete_model)

    def delete_model(self):
        reply = QtWidgets.QMessageBox.warning(self, 'Delete Model', 'Delete the saved model?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            shutil.rmtree(r'Models/')
            os.mkdir(r'Models')
            self.textEdit.setText("")
            QtWidgets.QMessageBox.information(self, 'Delete Done', 'You have deleted the model.')


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

    def save_results(self):
        res_info = self.textEdit.toPlainText()
        fname= QtWidgets.QFileDialog.getSaveFileName(directory='Results/')
        if fname[0]:
            res_save = open(fname[0], 'w')
            res_save.write(res_info)
            res_save.close()
            res_save.close()

    def start_training(self):
        # self.check_status()
        self.update_textedit("TestSetSamples:\n")
        QtCore.QCoreApplication.processEvents()
        #self.textEdit.append("Start training...\nTestSetSamples:\n")
        image_lists = MigrateTraining.create_image_lists(Parameters.TestSetPercentage, Parameters.ValidationPercentage)

        label_name_list = list(image_lists.keys())
        label_name_list.sort()
        for label_name in label_name_list:
            for sample in image_lists[label_name]['testing']:
                self.update_textedit(sample)
                QtCore.QCoreApplication.processEvents()

        n_classes = len(image_lists.keys())

        # 读取已经训练好的Inception-v3模型。
        with gfile.FastGFile(os.path.join(Parameters.MODEL_DIR, Parameters.MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def, return_elements=[Parameters.BOTTLENECK_TENSOR_NAME, Parameters.JPEG_DATA_TENSOR_NAME])

        # 定义新的神经网络输入
        bottleneck_input = tf.placeholder(tf.float32, [None, Parameters.BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

        # 定义一层全链接层
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(tf.truncated_normal([Parameters.BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
            biases = tf.Variable(tf.zeros([n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)

        # 定义交叉熵损失函数。
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(Parameters.LearningRate).minimize(cross_entropy_mean)

        # 计算正确率。
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            #初始化所有变量
            init = tf.global_variables_initializer()
            sess.run(init)

            #训练过程
            for i in range(Parameters.LearningSteps):
                self.progressBar.setValue(float(i) / Parameters.LearningSteps * 100.0 + 1.0)
                QtCore.QCoreApplication.processEvents()
                train_bottlenecks, train_ground_truth = MigrateTraining.get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, Parameters.BatchSize, 'training', jpeg_data_tensor, bottleneck_tensor)
                sess.run(train_step,
                         feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                if i % 100 == 0 or i + 1 == Parameters.LearningSteps:
                    validation_bottlenecks, validation_ground_truth = MigrateTraining.get_random_cached_bottlenecks(
                        sess, n_classes, image_lists, Parameters.BatchSize, 'validation', jpeg_data_tensor, bottleneck_tensor)
                    validation_accuracy = sess.run(evaluation_step, feed_dict={
                        bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                    mid_res_show = 'Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % \
                                   (i, Parameters.BatchSize, validation_accuracy * 100)
                    self.update_textedit(mid_res_show)
                    QtCore.QCoreApplication.processEvents()
                    saver.save(sess, os.path.join(Parameters.MODEL_SAVE_PATH, Parameters.MODEL_SAVE_NAME), global_step=i)

            # 在最后的测试数据上测试正确率。
            test_bottlenecks, test_ground_truth = MigrateTraining.get_test_bottlenecks(
                sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
            test_accuracy = sess.run(evaluation_step, feed_dict={
                bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
            res_show = 'Final test accuracy = %.1f%%' % (test_accuracy * 100)
            self.update_textedit(res_show)










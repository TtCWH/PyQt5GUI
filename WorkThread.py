from PyQt5 import QtCore, QtGui, QtWidgets
import time
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import Parameters
import PreProcess
import os.path

class TrainingProcess(QtCore.QThread):
    status_info = QtCore.pyqtSignal(str)
    done_percentage = QtCore.pyqtSignal(float)
    finish_signal = QtCore.pyqtSignal(bool)
    error_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(TrainingProcess, self).__init__()

    def run(self):
        #print(Parameters.INPUT_DATA)
        # self.check_status()
        #self.update_textedit()
        #QtCore.QCoreApplication.processEvents()
        #self.textEdit.append("Start training...\nTestSetSamples:\n")
        image_lists = PreProcess.create_image_lists(Parameters.TestSetPercentage, Parameters.ValidationPercentage)
        #print(image_lists)
        if image_lists == -1 or image_lists == {}:
            self.error_signal.emit(1)
            return

        label_name_list = list(image_lists.keys())
        label_name_list.sort()
        for label_name in label_name_list:
            for sample in image_lists[label_name]['testing']:
                self.status_info.emit(sample)
                #self.update_textedit(sample)
                #QtCore.QCoreApplication.processEvents()
        self.status_info.emit("TestSetSamples:")
        n_classes = len(image_lists.keys())
        Parameters.N_CLASSES = n_classes

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
                Done_percentage = float(i) / Parameters.LearningSteps * 100.0 + 1.0
                self.done_percentage.emit(Done_percentage)
                #QtCore.QCoreApplication.processEvents()

                try:
                    train_bottlenecks, train_ground_truth = PreProcess.get_random_cached_bottlenecks(
                        sess, n_classes, image_lists, Parameters.BatchSize, 'training', jpeg_data_tensor, bottleneck_tensor)
                except:
                    self.error_signal.emit(3)
                    return
                sess.run(train_step,
                        feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                if i % 100 == 0 or i + 1 == Parameters.LearningSteps:
                    try:
                        validation_bottlenecks, validation_ground_truth = PreProcess.get_random_cached_bottlenecks(
                            sess, n_classes, image_lists, Parameters.BatchSize, 'validation', jpeg_data_tensor, bottleneck_tensor)
                    except:
                        self.error_signal.emit(4)
                        return
                    validation_accuracy = sess.run(evaluation_step, feed_dict={
                        bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                    mid_res_show = 'Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % \
                                   (i, Parameters.BatchSize, validation_accuracy * 100)
                    self.status_info.emit(mid_res_show)
                    #self.update_textedit(mid_res_show)
                    #QtCore.QCoreApplication.processEvents()
                    saver.save(sess, os.path.join(Parameters.MODEL_SAVE_PATH, Parameters.MODEL_SAVE_NAME), global_step=i)

            # 在最后的测试数据上测试正确率。
            try:
                test_bottlenecks, test_ground_truth = PreProcess.get_test_bottlenecks(
                    sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
            except:
                self.error_signal.emit(5)
                return
            test_accuracy = sess.run(evaluation_step, feed_dict={
                bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
            res_show = 'Final test accuracy = %.1f%%' % (test_accuracy * 100)
            self.status_info.emit(res_show)
            #self.update_textedit(res_show)
            #QtCore.QCoreApplication.processEvents()

        self.finish_signal.emit(True)


class PicturePredict(QtCore.QThread):
    trigger = QtCore.pyqtSignal(int)
    finish_signal = QtCore.pyqtSignal(bool)
    error_signal = QtCore.pyqtSignal(int)

    def __init__(self,  image_path, parent=None):
        super(PicturePredict, self).__init__()
        self.image_Path = image_path

    def run(self):
        try:
            image_data = gfile.FastGFile(self.image_Path, 'rb').read()
            #print(image_data)
        except:
            self.error_signal.emit(2)
            return

        # 读取已经训练好的Inception-v3模型。
        with gfile.FastGFile(os.path.join(Parameters.MODEL_DIR, Parameters.MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def, return_elements=[Parameters.BOTTLENECK_TENSOR_NAME, Parameters.JPEG_DATA_TENSOR_NAME])

        bottleneck_input = tf.placeholder(tf.float32, [1, Parameters.BOTTLENECK_TENSOR_SIZE],
                                          name='BottleneckInputPlaceholder')

        # 定义一层全链接层
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(tf.truncated_normal([Parameters.BOTTLENECK_TENSOR_SIZE, Parameters.N_CLASSES], stddev=0.001))
            biases = tf.Variable(tf.zeros([Parameters.N_CLASSES]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(Parameters.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                try:
                    predict_bottleneck = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
                except:
                    self.error_signal.emit(2)
                    return
                # print(predict_bottleneck.shape)
                res = sess.run(final_tensor, feed_dict={bottleneck_input: predict_bottleneck})
                res = np.argmax(res, 1)
                # print(res)
                self.trigger.emit(res[0])
            else:
                print('No checkpoint file found.')
                self.trigger.emit(-1)

        self.finish_signal.emit(True)





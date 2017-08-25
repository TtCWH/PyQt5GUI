from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import tensorflow as tf
import Parameters
import PreProcess
import os.path
import shutil

class TrainingProcess(QtCore.QThread):
    status_info = QtCore.pyqtSignal(str)
    done_percentage = QtCore.pyqtSignal(float)
    finish_signal = QtCore.pyqtSignal(bool)
    error_signal = QtCore.pyqtSignal(int)
    my_model_save_path = Parameters.MODEL_SAVE_PATH
    my_model_save_name = Parameters.MODEL_SAVE_NAME

    def __init__(self, parent=None):
        super(TrainingProcess, self).__init__()
        self.runs = True

    def run(self):
        self.strat_training()
        self.stop()

    def stop(self):
        self.runs = False

    def strat_training(self):
        if self.runs:
            self.status_info.emit("Images Preprocessing")
            global my_model_save_name, my_model_save_path
            # print(Parameters.INPUT_DATA)
            # self.check_status()
            # self.update_textedit()
            # QtCore.QCoreApplication.processEvents()
            # self.textEdit.append("Start training...\nTestSetSamples:\n")
            image_lists, imageset_path = PreProcess.create_image_lists(Parameters.TestSetPercentage,
                                                                       Parameters.ValidationPercentage)
            if imageset_path:
                image_set_name = os.path.basename(imageset_path)
                my_model_save_name = image_set_name
                Parameters.MODEL_SAVE_NAME = my_model_save_name
                # print(image_set_name)
                my_model_save_path = os.path.join(Parameters.MODEL_SAVE_PATH, image_set_name)
                if not os.path.exists(my_model_save_path):
                    os.makedirs(my_model_save_path)

            # print(imageset_path)
            # print(image_lists)
            if image_lists == -1 or image_lists == {}:
                self.error_signal.emit(1)
                return

            # print(image_lists)
            label_name_list = list(image_lists.keys())
            label_name_list.sort()
            # print(label_name_list)
            Parameters.LABEL_NAME_LIST = label_name_list

            self.status_info.emit("TestSetSamples:")
            for label_name in label_name_list:
                for sample in image_lists[label_name]['testing']:
                    self.status_info.emit(label_name + ' ' + sample)
                    # self.update_textedit(sample)
                    # QtCore.QCoreApplication.processEvents()

            n_classes = len(image_lists.keys())
            Parameters.N_CLASSES = n_classes

            # 读取已经训练好的Inception-v3模型。
            with open(os.path.join(Parameters.MODEL_DIR, Parameters.MODEL_FILE), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def, return_elements=[Parameters.BOTTLENECK_TENSOR_NAME, Parameters.JPEG_DATA_TENSOR_NAME])

            # 定义新的神经网络输入
            bottleneck_input = tf.placeholder(tf.float32, [None, Parameters.BOTTLENECK_TENSOR_SIZE],
                                              name='BottleneckInputPlaceholder')
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
                # 初始化所有变量
                init = tf.global_variables_initializer()
                sess.run(init)

                # 训练过程
                for i in range(Parameters.LearningSteps):
                    Done_percentage = float(i) / Parameters.LearningSteps * 100.0
                    self.done_percentage.emit(Done_percentage)
                    # QtCore.QCoreApplication.processEvents()

                    try:
                        train_bottlenecks, train_ground_truth = PreProcess.get_random_cached_bottlenecks(
                            sess, n_classes, image_lists, imageset_path, Parameters.BatchSize, 'training',
                            jpeg_data_tensor, bottleneck_tensor)
                    except:
                        self.error_signal.emit(3)
                        return
                    sess.run(train_step,
                             feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                    if i % 100 == 0 or i + 1 == Parameters.LearningSteps:
                        try:
                            validation_bottlenecks, validation_ground_truth = PreProcess.get_random_cached_bottlenecks(
                                sess, n_classes, image_lists, imageset_path, Parameters.BatchSize, 'validation',
                                jpeg_data_tensor, bottleneck_tensor)
                        except:
                            self.error_signal.emit(4)
                            return
                        validation_accuracy = sess.run(evaluation_step, feed_dict={
                            bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                        mid_res_show = 'Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % \
                                       (i + 1, Parameters.BatchSize, validation_accuracy * 100)
                        self.status_info.emit(mid_res_show)
                        # self.update_textedit(mid_res_show)
                        # QtCore.QCoreApplication.processEvents()
                        saver.save(sess, os.path.join(my_model_save_path, my_model_save_name), global_step=i + 1)

                # 在最后的测试数据上测试正确率。
                try:
                    test_bottlenecks, test_ground_truth = PreProcess.get_test_bottlenecks(
                        sess, image_lists, imageset_path, n_classes, jpeg_data_tensor, bottleneck_tensor)
                except:
                    self.error_signal.emit(5)
                    return
                test_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
                res_show = 'Final test accuracy = %.1f%%' % (test_accuracy * 100)
                self.status_info.emit(res_show)
                self.done_percentage.emit(100)
                # self.update_textedit(res_show)
                # QtCore.QCoreApplication.processEvents()

            self.finish_signal.emit(True)


class PicturePredict(QtCore.QThread):
    trigger = QtCore.pyqtSignal(int)
    finish_signal = QtCore.pyqtSignal(bool)
    error_signal = QtCore.pyqtSignal(int)
    feed_back = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super(PicturePredict, self).__init__()
        self.runs = True

    def run(self):
        self.predict_picture()
        self.stop()

    def stop(self):
        self.runs = False

    def predict_picture(self):
        if self.runs:
            try:
                new_image = os.path.join(Parameters.TRAININGDATABASE, os.path.basename(Parameters.input_image_path))
                shutil.copyfile(Parameters.input_image_path, new_image)
                image_data = open(new_image, 'rb').read()
                # print(image_data)
            except:
                os.remove(new_image)
                self.error_signal.emit(2)
                return

            # 读取已经训练好的Inception-v3模型。
            with open(os.path.join(Parameters.MODEL_DIR, Parameters.MODEL_FILE), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def, return_elements=[Parameters.BOTTLENECK_TENSOR_NAME, Parameters.JPEG_DATA_TENSOR_NAME])

            bottleneck_input = tf.placeholder(tf.float32, [1, Parameters.BOTTLENECK_TENSOR_SIZE],
                                              name='BottleneckInputPlaceholder')

            # 定义一层全链接层
            with tf.name_scope('final_training_ops'):
                weights = tf.Variable(
                    tf.truncated_normal([Parameters.BOTTLENECK_TENSOR_SIZE, Parameters.N_CLASSES], stddev=0.001))
                biases = tf.Variable(tf.zeros([Parameters.N_CLASSES]))
                logits = tf.matmul(bottleneck_input, weights) + biases
                final_tensor = tf.nn.softmax(logits)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                predict_model_path = Parameters.MODEL_SAVE_PATH + '/' + Parameters.MODEL_SAVE_NAME
                ckpt = tf.train.get_checkpoint_state(predict_model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    try:
                        predict_bottleneck = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
                        # print(predict_bottleneck)
                    except:
                        # print('WANG')
                        if os.path.exists(new_image):
                            os.remove(new_image)
                        self.error_signal.emit(6)
                        return
                    # print(predict_bottleneck.shape)
                    res = sess.run(final_tensor, feed_dict={bottleneck_input: predict_bottleneck})
                    res = np.argmax(res, 1)
                    # print(res)
                    self.trigger.emit(res[0])
                    self.feed_back.emit([res[0], new_image, Parameters.LABEL_NAME_LIST])
                else:
                    print('No checkpoint file found.')
                    self.trigger.emit(-1)
            self.finish_signal.emit(True)










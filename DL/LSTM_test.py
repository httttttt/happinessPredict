import os
import writeCSV
import readData
import datetime
import xgboost_feature
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定 gtx1060 6g
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def lstm(x_train, x_test, y_train, y_test):
    num, col = x_train.shape
    _, num_class = y_train.shape
    # 参数设置
    BATCH_SIZE = 1  # BATCH的大小，相当于一次处理1个样本
    TIME_STEP = 1  # 一个LSTM中，输入序列的长度，时间序列的一个样本只有1行
    INPUT_SIZE = col  # x_i 的向量长度，样本有20001列
    LR = 0.1  # 学习率
    NUM_UNITS = 1  # 多少个LTSM单元
    ITERATIONS = 2000  # 迭代次数
    LEARNING_RATE_DECAY = 0.99
    MOVING_AVERAGE_DECAY = 0.99

    N_CLASSES = num_class  # 输出大小，每一个类别的概率，例如：[1, 0, 0, 0]

    # 定义 placeholders 以便接收x,y   输入的可以是一张图像展开成的一维向量
    # 维度是[BATCH_SIZE，TIME_STEP * INPUT_SIZE]
    train_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])
    # 输入的是二维数据，将其还原为三维，维度是[BATCH_SIZE, TIME_STEP, INPUT_SIZE]
    image = tf.reshape(train_x, [-1, TIME_STEP, INPUT_SIZE])
    train_y = tf.placeholder(tf.int32, [None, N_CLASSES])

    # 定义RNN（LSTM）结构
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS)
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=NUM_UNITS)
    outputs, final_state = tf.nn.dynamic_rnn(
        cell=rnn_cell,          # 选择传入的cell
        inputs=image,           # 传入的数据
        initial_state=None,     # 初始状态
        dtype=tf.float32)       # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
    # 输出层 输出的是每一个类别的概率，概率最大的就判定为是该类
    # dense 连接一个全连接层, 取最后一个时刻的循环网络的输出作为全连接层的输入
    output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_CLASSES)

    # 定义交叉熵 损失函数
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_y, logits=output)  # 计算loss
    loss = tf.losses.mean_squared_error(labels=train_y, predictions=output)
    # 定义用于滑动平均训练轮数的变量，这个变量并不需要滑动平均
    global_step = tf.Variable(0, trainable=False)
    # 定义滑动平均的类
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 对所有变量进行滑动平均-->当前图上所有可训练的变量
    variables_averages_op = variables_averages.apply(tf.trainable_variables())
    # 设置指数衰减的学习速率
    learning_rate = tf.train.exponential_decay(LR, global_step, num / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)  # 选择优化方法
    train_op = tf.group(train_step, variables_averages_op)

    # 设置GPU占用率
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())  # 初始化计算图中的变量

    x = x_train
    y = y_train
    test_x = x_test
    test_y = y_test
    for step in range(ITERATIONS):  # 开始训练
        _, loss_ = sess.run([train_op, loss], feed_dict={train_x: x, train_y: y})
        if step % 100 == 0:  # test（validation）
            print('after training %i step, train loss: %.3f' % (step, loss_))
            # true_y = sess.run(tf.argmax(test_y, axis=1))
            print('true is: ', test_y)
            predict = sess.run(output, feed_dict={train_x: test_x})
            prediction = sess.run(tf.argmax(predict, axis=1))
            print('predict is: ', predict)
            print('**********' * 10)
    print('finished!')
    return prediction


def test1():
    trainFeature, happiness, testFeature = xgboost_feature.get_xgb_feature()
    x_train, x_test, y_train, y_test = train_test_split(trainFeature, happiness, test_size=0.333)
    y_train_array = np.zeros((len(y_train), 5))
    for i in range(len(y_train)):
        if y_train[i] == 1:
            y_train_array[i] = np.array([1, 0, 0, 0, 0])
        elif y_train[i] == 2:
            y_train_array[i] = np.array([0, 1, 0, 0, 0])
        elif y_train[i] == 3:
            y_train_array[i] = np.array([0, 0, 1, 0, 0])
        elif y_train[i] == 4:
            y_train_array[i] = np.array([0, 0, 0, 1, 0])
        elif y_train[i] == 5:
            y_train_array[i] = np.array([0, 0, 0, 0, 1])

    y_test_array = np.zeros((len(y_test), 5))
    for i in range(len(y_test)):
        if y_test[i] == 1:
            y_test_array[i] = np.array([1, 0, 0, 0, 0])
        elif y_test[i] == 2:
            y_test_array[i] = np.array([0, 1, 0, 0, 0])
        elif y_test[i] == 3:
            y_test_array[i] = np.array([0, 0, 1, 0, 0])
        elif y_test[i] == 4:
            y_test_array[i] = np.array([0, 0, 0, 1, 0])
        elif y_test[i] == 5:
            y_test_array[i] = np.array([0, 0, 0, 0, 1])
    prediction = lstm(x_train, x_test, y_train_array, y_test_array)
    return prediction


if __name__ ==  '__main__':
    """
    不收敛
    """
    trainFeature, happiness, testFeature = xgboost_feature.get_xgb_feature()
    x_train, x_test, y_train, y_test = train_test_split(trainFeature, happiness, test_size=0.333)

    # prediction = test1()
    # mse = mean_squared_error(np.argmax(y_test_array, axis=1), prediction)
    # print(mse)
    # path = 'E:\python_project\happinessPredict\submit_lstm.csv'
    # writeCSV.writCSV(path, prediction)

    prediction = lstm(x_train, x_test, y_train, y_test)
    print(prediction)

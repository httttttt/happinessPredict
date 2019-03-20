import os
import xgboost
import readData
import numpy as np
import xgboost_feature
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定 gtx1060 6g


INPUT_NODE = 39
OUTPUT_NODE = 5
LAYER1_NODE = 500
BATCH_SIZE = 1

LEARNING_RATE_BASE = 1.0
LEARNING_RATE_DECAY = 0.99
REGULARIZATIONG_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        # 不使用滑动平均的前向传播
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.nn.softmax(tf.matmul(layer1, weights2) + biases2)
    else:
        # 使用滑动平均的前向传播，首先对变量计算滑动平均
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.nn.softmax(tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2))


# 模型训练过程
def train(x_train, x_test, y_train, y_test):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 训练集个数
    num, _ = x_train.shape

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 进行前向传播
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义用于滑动平均训练轮数的变量，这个变量并不需要滑动平均
    global_step = tf.Variable(0, trainable=False)
    # 定义滑动平均的类
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 对BP神经网络的所有变量进行滑动平均-->当前图上所有可训练的变量
    variables_averages_op = variables_averages.apply(tf.trainable_variables())
    average_y = inference(x, variables_averages, weights1, biases1, weights2, biases2)

    # 交叉熵作为损失函数
    mse = tf.losses.mean_squared_error(labels=tf.argmax(y_, 1), predictions=tf.argmax(average_y, 1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=average_y, labels=tf.argmax(y_, 1))
    # # 计算交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATIONG_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习速率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, num / BATCH_SIZE, LEARNING_RATE_DECAY)

    # 使用梯度下降来优化参数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # train_step 和 variables_averages_op 操作同时进行
    train_op = tf.group(train_step, variables_averages_op)

    # 查看滑动平均下的准确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 设置GPU占用率
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # 初始化所有参数
        tf.global_variables_initializer().run()
        # 测试集数据
        test_feed = {x: x_test, y_: y_test}
        MSEs = []
        for i in range(TRAINING_STEPS):
            sess.run(train_op, feed_dict={x: x_train, y_: y_train})
            if i % 100 == 0:
                predict = sess.run(average_y, feed_dict={x: x_test})
                prediction = sess.run(tf.argmax(predict, axis=1))
                print(prediction)
                print(sess.run(tf.argmax(y_test, axis=1)))
                print('***********' * 10)
                test_mse = sess.run(accuracy, feed_dict=test_feed)
                MSEs.append(test_mse)
                print("after %d training steps, test_mse using average model is %g" % (i, test_mse))
    return MSEs


# 主程序入口
def main(argv=None):
    """
    不收敛
    """
    num_repeat = 1
    MSEs = []
    abbrTrain = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    for _ in range(num_repeat):
        x_train, x_test, y_train, y_test = readData.readData(abbrTrain, True)
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
        # 对每一行的样本的同一位置的特征进行z-score标准化
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        mse = train(x_train, x_test, y_train_array, y_test_array)
        MSEs.append(mse)
        tf.reset_default_graph()

    plt.figure()
    plt.grid()
    plt.xlabel('iteration$(\\times10^2)$')
    plt.ylabel('MSE')
    # plt.axis([0, len(mse), 0, 1.1])
    for i in range(num_repeat):
        plt.plot(MSEs[i])
    # 保存图片
    # plt.savefig("diff_s1423 with PCA+ANN.svg", transparent=True, format='svg')
    plt.show()


if __name__ == '__main__':
    tf.app.run()



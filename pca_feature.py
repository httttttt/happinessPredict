import readData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def get_x_ratio(data, target_ratio):
    """
    对输入data提取占比为target_ratio的前m个特征

    svd_solver：即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。
    有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。
    randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。
    full则是传统意义上的SVD，使用了scipy库对应的实现。
    arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，
    而arpack直接使用了scipy库的sparse SVD实现。
    默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。

    :param data: 待降维数据（每一行为一个样本）
    :param target_ratio: 前多少百分比
    :return: pca类，主成分分析后的特征向量的映射（已经经过data * 前多少个特征向量）
    """
    row, col = data.shape
    if row < col:
        components = row
    else:
        components = col
    pca = PCA(n_components=components, svd_solver='auto')
    pca_data = pca.fit_transform(data)
    total_list = pca.explained_variance_ratio_.tolist()
    total = 0
    for i in range(len(total_list)):
        total += total_list[i]
        if total >= target_ratio:
            pca = PCA(n_components=i+1, svd_solver='auto')
            pca_data = pca.fit_transform(data)
            return pca, pca_data


if __name__ == '__main__':
    abbrTrain = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    abbrTest = 'E:\python_project\happinessPredict\DataSet\happiness_test_abbr.csv'
    # trainData, happiness = readData.readData(abbrTrain, True)
    # # 对每一行的样本的同一位置的特征进行z-score标准化
    # trainData = preprocessing.scale(trainData)
    # print(trainData.mean(axis=0))
    # print(trainData.std(axis=0))

    trainData, happiness = readData.readData(abbrTrain, True)
    scaler = preprocessing.StandardScaler().fit(trainData)
    trainData = scaler.transform(trainData)

    pca, pca_data = get_x_ratio(trainData, 0.95)

    testData = readData.readData(abbrTest, False)
    testData = scaler.transform(testData)

    # 绘制主成分直方图
    ratio = pca.explained_variance_ratio_.tolist()
    plt.figure()
    plt.grid()
    plt.bar(range(len(ratio)), ratio, alpha=0.9, facecolor="lightskyblue", edgecolor="white", width=0.5)
    plt.title('Principal Component Analysis', fontsize=9)
    plt.xlabel('Eigenvalues', fontsize=9)
    plt.ylabel('Contribution rate of eigenvalues', fontsize=9)
    number = []
    for i in range(len(ratio)):
        number.append(i)

    plt.xticks(number, rotation=-30)  # rotation控制倾斜角度
    # 绘制bar值
    for _x, _y in zip(range(len(ratio)), ratio):
        plt.text(_x, _y, '%.3f' % _y, ha='center', va='bottom', size=9)

    # plt.savefig("pca.svg", transparent=True, format='svg')
    plt.show()

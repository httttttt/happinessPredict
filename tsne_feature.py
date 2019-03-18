import readData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'


if __name__ == '__main__':
    abbrTrain = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    abbrTest = 'E:\python_project\happinessPredict\DataSet\happiness_test_abbr.csv'
    trainData, happiness = readData.readData(abbrTrain, True)
    trainData = np.nan_to_num(trainData)

    tsne = TSNE(n_components=2, init='pca')
    # targetData = tsne.fit_transform(trainData)
    # plt.figure()
    # print(happiness[0])

    # fig_s1423 = plt.scatter(targetData[:39, 0], targetData[:39, 1], marker='o')
    # fig_s1423_1s27 = plt.scatter(targetData[40:79, 0], targetData[40:79, 1], marker='s')
    # fig_s1423_2s27 = plt.scatter(targetData[80:119, 0], targetData[80:119, 1], marker='+')
    # fig_s1423_3s27 = plt.scatter(targetData[120:, 0], targetData[120:, 1], marker='*')
    # plt.legend((fig_s1423, fig_s1423_1s27, fig_s1423_2s27, fig_s1423_3s27),
    #            ('s14237', 's1423_1s27', 's1423_2s27', 's1423_3s27'))
    # plt.grid()
    # plt.show()


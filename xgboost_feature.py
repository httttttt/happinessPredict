import readData
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def extract_feature(xgbClass, data, rate):
    importance = xgbClass.feature_importances_
    colums = data.columns
    newImportance = np.sort(-importance)
    newImportance = -newImportance
    index = np.argsort(-importance)
    # 取出占比为前rate的数据
    ans = 0
    for i in range(len(newImportance)):
        ans += newImportance[i]
        if ans >= rate:
            targetCol = colums[index[:i]]
            break
    # 提取index对应的特征
    targetData = [data[str(targetCol[i])] for i in range(len(targetCol))]
    return targetData, targetCol


if __name__ == '__main__':
    abbrTrain = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    trainData, happiness = readData.readData(abbrTrain, True)
    # # 对每一行的样本的同一位置的特征进行z-score标准化
    # scaler = preprocessing.StandardScaler().fit(trainData)
    # trainData = scaler.transform(trainData)

    x_train, x_test, y_train, y_test = train_test_split(trainData, happiness, test_size=0.3)
    # xgboost 自带的分类器
    classifier = xgb.XGBClassifier()
    classifier.fit(x_train, y_train)
    xgb.plot_importance(classifier)
    plt.show()
    MSE_classifier = mean_squared_error(classifier.predict(x_test), y_test)

    # xgboost 自带的回归器
    regressor = xgb.XGBRegressor()
    regressor.fit(x_train, y_train)
    xgb.plot_importance(regressor)
    plt.show()
    MSE_regressor = mean_squared_error(regressor.predict(x_test), y_test)

    # targetFeature, targetCol = extract_feature(classifier, trainData, 0.90)
    # print('targetCol is:', targetCol)
    abbrTest = 'E:\python_project\happinessPredict\DataSet\happiness_test_abbr.csv'
    testData = readData.readData(abbrTest, False)
    # testData = scaler.transform(testData)

    if MSE_regressor < MSE_classifier:
        print('this is a regress problem, MSE=%.3f' % MSE_regressor)
        prediction = regressor.predict(testData)
        print(prediction)
    else:
        print('this is a classify problem, MSE=%.3f' % MSE_classifier)
        prediction = classifier.predict(testData)
        print(prediction)

"""
结论：
是否进行标准化对xgboost选取的特征影响不大
"""

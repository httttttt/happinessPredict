import csv
import readData
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def extract_feature(xgb, data, rate):
    importance = xgb.feature_importances_
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
    # 共23个特征，6400个样本
    targetData = np.nan_to_num(targetData).T
    return targetData, targetCol


def get_xgb_feature():
    abbrTrain = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    x_train, x_test, y_train, y_test, trainData, happiness = readData.xgb_readData(abbrTrain, True)
    regressor = xgb.XGBRegressor().fit(x_train, y_train)
    trainFeature, trainCol = extract_feature(regressor, trainData, 0.90)
    happiness = np.nan_to_num(happiness)
    # print('targetFeature is:', targetFeature)
    # print('targetCol is:', targetCol)
    abbrTest = 'E:\python_project\happinessPredict\DataSet\happiness_test_abbr.csv'
    testData = readData.xgb_readData(abbrTest, False)
    testFeature, testCol = extract_feature(regressor, testData, 0.90)
    return trainFeature, happiness, testFeature


if __name__ == '__main__':
    abbrTrain = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    x_train, x_test, y_train, y_test, trainData, happiness = readData.xgb_readData(abbrTrain, True)
    # # 对每一行的样本的同一位置的特征进行z-score标准化
    # scaler = preprocessing.StandardScaler().fit(trainData)
    # trainData = scaler.transform(trainData)

    # xgboost 自带的分类器
    classifier = xgb.XGBClassifier().fit(x_train, y_train)
    xgb.plot_importance(classifier)
    plt.show()
    MSE_classifier = mean_squared_error(classifier.predict(x_test), y_test)

    # xgboost 自带的回归器
    regressor = xgb.XGBRegressor().fit(x_train, y_train)
    xgb.plot_importance(regressor)
    plt.show()
    MSE_regressor = mean_squared_error(regressor.predict(x_test), y_test)

    abbrTest = 'E:\python_project\happinessPredict\DataSet\happiness_test_abbr.csv'
    testData = readData.xgb_readData(abbrTest, False)

    if MSE_regressor < MSE_classifier:
        print('this is a regress problem, MSE=%.3f' % MSE_regressor)
        prediction = regressor.predict(testData)
        prediction = prediction.tolist()
        # with open('E:\python_project\happinessPredict\submit.csv', 'w', newline='') as csvFile:
        #     fileNames = ['id', 'happiness']
        #     writer = csv.DictWriter(csvFile, fieldnames=fileNames)
        #     writer.writeheader()
        #     for i in range(len(prediction)):
        #         dictionary = {'id': i+8001, 'happiness': prediction[i]}
        #         writer.writerow(dictionary)
    else:
        print('this is a classify problem, MSE=%.3f' % MSE_classifier)
        prediction = classifier.predict(testData)
        print(prediction)
        # with open('E:\python_project\happinessPredict\submit.csv', 'w') as csvFile:
        #     writer = csv.writer(csvFile)
        #     writer.writerow(['happiness'])
        #     writer.writerow(prediction.tolist)


"""
结论：
是否进行标准化对xgboost选取的特征影响不大
"""

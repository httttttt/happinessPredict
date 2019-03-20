import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def xgb_readData(path, trainingFlag):
    if trainingFlag == True:
        col1 = [i+2 for i in range(4)]
        col2 = [i+7 for i in range(35)]
        col = col1 + col2
        trainData = pd.read_csv(path, usecols=col)
        happiness = pd.read_csv(path, usecols=[1])
        x_train, x_test, y_train, y_test = train_test_split(trainData, happiness, test_size=0.2)
        return x_train, x_test, y_train, y_test, trainData, happiness
    elif trainingFlag == False:
        col1 = [i+1 for i in range(4)]
        col2 = [i+6 for i in range(35)]
        col = col1 + col2
        testData = pd.read_csv(path, usecols=col)
        return testData


def readData(path, trainingFlag):
    if trainingFlag == True:
        col1 = [i+2 for i in range(4)]
        col2 = [i+7 for i in range(35)]
        col = col1 + col2
        trainData = pd.read_csv(path, usecols=col)
        happiness = pd.read_csv(path, usecols=[1])
        trainData = np.nan_to_num(trainData)
        happiness = np.nan_to_num(happiness)
        x_train, x_test, y_train, y_test = train_test_split(trainData, happiness, test_size=0.2)
        return x_train, x_test, y_train, y_test
    elif trainingFlag == False:
        col1 = [i+1 for i in range(4)]
        col2 = [i+6 for i in range(35)]
        col = col1 + col2
        testData = pd.read_csv(path, usecols=col)
        testData = np.nan_to_num(testData)
        return testData


if __name__ == '__main__':
    abbrTrain = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    abbrTest = 'E:\python_project\happinessPredict\DataSet\happiness_test_abbr.csv'
    trainData, happiness = readData(abbrTrain, True)
    print(trainData.shape)

    testData = readData(abbrTest, False)
    print(testData.shape)

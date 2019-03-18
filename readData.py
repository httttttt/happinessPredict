import pandas as pd
from sklearn.model_selection import train_test_split


def readData():
    col1 = [i + 2 for i in range(4)]
    col2 = [i + 7 for i in range(35)]
    col = col1 + col2
    abbrPath = 'E:\python_project\happinessPredict\DataSet\happiness_train_abbr.csv'
    data = pd.read_csv(abbrPath, usecols=col)
    happiness = pd.read_csv(abbrPath, usecols=[1])

    return data, happiness


if __name__ == '__main__':
    data, happiness = readData()
    print(happiness)

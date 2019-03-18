import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel


if __name__ == '__main__':
    col1 = [i+2 for i in range(4)]
    col2 = [i+7 for i in range(35)]
    col = col1 + col2
    # print(col)

    abbrPath = 'E:\python_project\workingspace\TianChi_Match\Happyness\DataSet\happiness_train_abbr.csv'
    data = pd.read_csv(abbrPath, usecols=col)
    happiness = pd.read_csv(abbrPath, usecols=[1])


    """
    设置训练集和测试集
    """
    x_train, x_test, y_train, y_test = train_test_split(data, happiness, test_size=0.3)
    # # xgboost 自带 分类器 和 回归器 类
    # classifier = xgb.XGBClassifier()
    # classifier.fit(x_train, y_train)
    # xgb.plot_importance(classifier)
    # plt.show()
    # print(classifier.feature_importances_)

    regressor = xgb.XGBRegressor()
    regressor.fit(data, happiness)
    xgb.plot_importance(regressor)
    plt.show()

    importance = regressor.feature_importances_
    colums = data.columns
    newImportance = np.sort(-importance)
    newImportance = -newImportance
    index = np.argsort(-importance)
    # 取出占比为前0.6的数据
    ans = 0
    for i in range(len(newImportance)):
        ans += newImportance[i]
        if ans >= 0.6:
            targetCol = colums[index[:i]]
            break
    #
    for i in range(len(targetCol)):
        print(data[str(targetCol[i])])

import numpy as np
import xgboost_feature
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def SVM_classifier():
    trainFeature, happiness, testFeature = xgboost_feature.get_xgb_feature()
    x_train, x_test, y_train, y_test = train_test_split(trainFeature, happiness, test_size=0.333)

    poly = SVC(kernel='poly', C=10, gamma=10, degree=120, coef0=1, max_iter=12000).fit(x_train, y_train.ravel())
    prediction = poly.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=prediction)
    print(mse)


def SVM_regressor():
    trainFeature, happiness, testFeature = xgboost_feature.get_xgb_feature()
    x_train, x_test, y_train, y_test = train_test_split(trainFeature, happiness, test_size=0.333)
    regressor = SVR(C=10, gamma=10, degree=120, coef0=1, max_iter=12000).fit(x_train, y_train.ravel())
    prediction = regressor.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=prediction)
    print(prediction)
    print(mse)


if __name__ == '__main__':
    SVM_regressor()

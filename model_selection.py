import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def cross_validation(fold=10, shuffle=True):
    data = pd.read_csv('./data/thyroid_clean.csv')
    data = data.drop(columns='id')
    y = data['mal'].values
    x = data.drop(columns='mal').values
    index0 = np.where(y == 0)[0]
    index1 = np.where(y == 1)[0]
    if shuffle:
        np.random.shuffle(index0)
        np.random.shuffle(index1)

    step0 = int(index0.shape[0] / fold)
    step1 = int(index1.shape[0] / fold)
    for i in range(fold):
        start0 = step0 * i
        end0 = start0 + step0
        start1 = step1 * i
        end1 = start1 + step1
        if i == fold - 1:
            end0 = index0.shape[0]
            end1 = index1.shape[0]

        test_set_boolean = np.zeros(y.shape[0], dtype='bool')
        test_set_boolean[index0[start0:end0]] = True
        test_set_boolean[index1[start1:end1]] = True
        x_test = x[test_set_boolean]
        y_test = y[test_set_boolean]
        x_train = x[~test_set_boolean]
        y_train = y[~test_set_boolean]

        yield x_train, y_train, x_test, y_test


if __name__ == '__main__':
    cross_validation()

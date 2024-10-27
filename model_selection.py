import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


def one_hot(arr, num_classes=2):
    labels = np.zeros((arr.shape[0], num_classes), dtype=arr.dtype)
    for i, v in enumerate(arr):
        labels[i, v] = 1

    return labels


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


def accuracy_score(label_true, label_pred):
    return metrics.accuracy_score(label_true, label_pred)


def auc_score(y_true_proba, y_pred_proba):
    return metrics.roc_auc_score(y_true_proba, y_pred_proba)


def sensitivity_score(label_true, label_pred):
    return metrics.recall_score(label_true, label_pred)


def specificity_score(label_true, label_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(label_true, label_pred).ravel()
    return tn / (tn + fp)


def precision_score(label_true, label_pred):
    return metrics.precision_score(label_true, label_pred)


def train_and_evaluate(model):
    acc_list, auc_list, sen_list, spe_list, prec_list = [], [], [], [], []

    for x_train, y_train, x_test, y_test in cross_validation():
        model.fit(x_train, y_train)
        y_pred_proba = model.predict_proba(x_test)
        y_pred = y_pred_proba.argmax(axis=1)
        y_test_proba = one_hot(y_test, num_classes=2)

        acc_list.append(accuracy_score(y_test, y_pred))
        auc_list.append(auc_score(y_test_proba, y_pred_proba))
        sen_list.append(sensitivity_score(y_test, y_pred))
        spe_list.append(specificity_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))

    return np.mean(acc_list), np.mean(auc_list), np.mean(sen_list), np.mean(spe_list), np.mean(prec_list)


if __name__ == '__main__':
    model_dict = {
        '逻辑回归': LogisticRegression(solver='liblinear'),
        '随机森林': RandomForestClassifier(),
        '线性判别分析': LinearDiscriminantAnalysis(),
        '梯度提升机': GradientBoostingClassifier(),
        '支持向量机': SVC(kernel='linear', probability=True)
    }
    for model_name, model in model_dict.items():
        acc, auc, sen, spe, prec = train_and_evaluate(model)
        print('{}: acc={:.4f}, auc={:.4f}, sen={:.4f}, spe={:.4f}, prec={:.4f}'.format(
            model_name, acc, auc, sen, spe, prec
        ))

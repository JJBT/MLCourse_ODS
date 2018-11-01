import warnings
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, \
    roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import fetch_20newsgroups, load_files

from scipy.sparse import csr_matrix

plt.style.use('ggplot')
warnings.filterwarnings('ignore')


df = pd.read_csv('dataset/bank_train.csv')
labels = pd.read_csv('dataset/bank_train_target.csv', header=None)

df['education'].value_counts().plot.barh()
plt.show()


def logistic_regression_accuracy_on(dataframe, labels):
    features = dataframe.as_matrix()
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels)

    logit = LogisticRegression()
    logit.fit(train_features, train_labels)
    print('Accuracy', accuracy_score(test_labels, logit.predict(test_features)))
    return classification_report(test_labels, logit.predict(test_features))


label_en = LabelEncoder()

categorical_columns = df.columns[df.dtypes == 'object']
for column in categorical_columns:
    df[column] = label_en.fit_transform(df[column])

print(logistic_regression_accuracy_on(df[categorical_columns], labels))

"""Такой подход создает метрику сходства. Не подходит для линейных моделей"""

onehot_encoder = OneHotEncoder(sparse=False)
encoded_categ_col = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_columns]))

print(logistic_regression_accuracy_on(encoded_categ_col, labels))


"""Hashing tricks"""

hash_space = 25

hashing_example = pd.DataFrame([{i: 0.0 for i in range(hash_space)}])
for s in ('job=student', 'marital=single', 'day_of_week=mon'):
    print(s, '->', hash(s) % hash_space)
    hashing_example.loc[0, hash(s) % hash_space] = 1



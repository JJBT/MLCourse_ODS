"""Пример использования PCA"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


iris = datasets.load_iris()
X = iris.data
y = iris.target


# Decision Tree with source data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print('Tree Accuracy {:.5f}'.format(accuracy_score(y_test, preds)))


# Decision tree with PCA
pca = PCA(n_components=2)
X_centered = X - X.mean(axis=0)
X_pca = pca.fit_transform(X_centered)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, stratify=y, random_state=42)

clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print('Tree PCA accuracy {:.5f}'.format(accuracy_score(y_test, preds)))

"""Распознавание рукописных цифр"""

import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = load_digits()
X, y = data['data'], data['target']

X[0, :].reshape([8, 8])

# f, axes = plt.subplots(1, 4, sharey=True, figsize=(16, 6))
#
# for i in range(4):
#     axes[i].imshow(X[i, :].reshape([8, 8]))

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=42)

"""Обучим дерево решений и kNN, опять параметры пока наугад берем."""
tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)

tree.fit(X_train, y_train)
knn.fit(X_train, y_train)

tree_pred = tree.predict(X_holdout)
knn_pred = knn.predict(X_holdout)
as1 = accuracy_score(y_holdout, tree_pred), accuracy_score(y_holdout, knn_pred)

tree_params = {'max_depth': [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],
               'max_features': [1, 2, 3, 5, 10, 20, 30, 50, 64]}
tree_grid = GridSearchCV(tree, tree_params, cv=5, verbose=4)

tree_grid.fit(X_train, y_train)

print(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv=5)))


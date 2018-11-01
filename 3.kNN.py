import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


"""Первую модель будем обучать без штатов, потом посмотрим, помогают ли они."""
df = pd.read_csv('dataset/telecom_churn.csv')

df['International plan'] = pd.factorize(df['International plan'])[0]
df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
df['Churn'] = df['Churn'].astype('int')
states = df['State']
y = df['Churn']
df.drop(['State', 'Churn'], axis=1, inplace=True)

"""Выделим 70% выборки (X_train, y_train) под обучение и 30% будут отложенной выборкой (X_holdout, y_holdout)"""
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3, random_state=17)

tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)

tree.fit(X_train, y_train)
knn.fit(X_train, y_train)

"""Качество - доля правильных ответов"""
tree_pred = tree.predict(X_holdout)
as1 = accuracy_score(y_holdout, tree_pred)

knn_pred = knn.predict(X_holdout)
as2 = accuracy_score(y_holdout, knn_pred)

"""Теперь настроим параметры дерева на кросс-валидации. Настраивать будем максимальную глубину
и максимальное используемое на каждом разбиении число признаков.
Суть того, как работает GridSearchCV: для каждой уникальной пары значений параметров max_depth и max_features
будет проведена 5-кратная кросс-валидация и выберется лучшее сочетание параметров"""

tree_params = {'max_depth': range(1, 11),
               'max_features': range(4, 19)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, verbose=True)
tree_grid.fit(X_train, y_train)

# print(tree_grid.best_params_)  # Лучшее сочетание параметров

as3 = accuracy_score(y_holdout, tree_grid.predict(X_holdout))

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_pipe_params = {'knn__n_neighbors': range(1, 10),
                   'knn__p': [1, 2, 3, 1000]}

knn_grid = GridSearchCV(knn_pipe, knn_pipe_params, cv=5, verbose=True)

knn_grid.fit(X_train, y_train)

as4 = accuracy_score(y_holdout, knn_grid.predict(X_holdout))

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=17)
print(np.mean(cross_val_score(forest, X_train, y_train, cv=5)))

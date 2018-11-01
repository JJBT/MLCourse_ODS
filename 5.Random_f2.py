import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/telecom_churn.csv')

# Выбираем сначала только колонки с числовым типом данных
cols = []
for i in df.columns:
    if (df[i].dtype == 'float64') or (df[i].dtype == 'int64'):
        cols.append(i)

# Разделяем на признаки и объекты
X, y = df[cols].copy(), np.asarray(df['Churn'], dtype='int8')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfc = RandomForestClassifier(random_state=42, oob_score=True)

results = cross_val_score(rfc, X, y, cv=skf)

print('CV accuracy score: {:.2f} %'.format(results.mean()*100))


"""Кривые валидации по подбору кол-ва деревьев"""

grid = [5, 10, 15, 20, 30, 50, 75, 100]
rf = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1)

val_train, val_test = validation_curve(rf, X, y, 'n_estimators', grid, cv=skf,
                                       scoring='accuracy')


def plot_func(x, data):
    mean, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mean, 'o-')


plot_func(grid, val_train)
plot_func(grid, val_test)
plt.show()


"""Кривые валидации по подбору максимальной глубины. Зафиксируем кол-во деревьев = 100"""

max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)

val_train, val_test = validation_curve(rf, X, y, 'max_depth', max_depth_grid, cv=skf,
                                       scoring='accuracy')

plot_func(max_depth_grid, val_train)
plot_func(max_depth_grid, val_test)
plt.show()


"""Кривые валидации по подбору min_samples_leaf. Зафиксируем кол-во деревьев = 100"""

min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)

val_train, val_test = validation_curve(rf, X, y, 'min_samples_leaf', min_samples_leaf_grid, cv=skf,
                                       scoring='accuracy')

plot_func(min_samples_leaf_grid, val_train)
plot_func(min_samples_leaf_grid, val_test)
plt.show()


"""Кривые валидации по подбору max_features. Зафиксируем кол-во деревьев = 100"""

max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]

rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)

val_train, val_test = validation_curve(rf, X, y, 'max_features', max_features_grid, cv=skf,
                                       scoring='accuracy')

plot_func(max_features_grid, val_train)
plot_func(max_features_grid, val_test)
plt.show()


"""Поиск оптимальных параметров"""

parameters = {'max_features': [4, 7, 10, 13], 'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5, 10, 15, 20]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
gcv = GridSearchCV(rfc, parameters, cv=skf)
gcv.fit(X, y)


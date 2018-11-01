from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('dataset/microchip_tests.txt', header=None, names=('test1', 'test2', 'released'))

X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
plt.xlabel("Тест 1")
plt.ylabel("Тест 2")
plt.title('2 теста микрочипов')
plt.legend()
plt.grid()


def plot_boundary(clf, X, grid_step=.01, poly_featurizer=None):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # каждой точке в сетке [x_min, x_max]x[y_min, y_max]
    # ставим в соответствие свой цвет
    if poly_featurizer:
        Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap='Blues')


poly = PolynomialFeatures(degree=7)
X_poly = poly.fit_transform(X)

C = 1e-2
logit = LogisticRegression(C=C, random_state=17)
logit.fit(X_poly, y)

# plot_boundary(logit, X, grid_step=.01, poly_featurizer=poly)
#
# plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
# plt.xlabel("Тест 1")
# plt.ylabel("Тест 2")
# plt.title('2 теста микрочипов. Логит с C=0.01')
# plt.legend()

print("Доля правильных ответов классификатора на обучающей выборке:", round(logit.score(X_poly, y), 3))

""" Видим, что регуляризация оказалась слишком сильной, и модель "недообучилась"""


"""Увеличим C до 1. Тем самым мы ослабляем регуляризацию, теперь в решении значения весов логистической регрессии
 могут оказаться больше (по модулю), чем в прошлом случае."""

C = 1
logit = LogisticRegression(C=C, random_state=17)
logit.fit(X_poly, y)

plot_boundary(logit, X, grid_step=.005, poly_featurizer=poly)

plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
plt.xlabel("Тест 1")
plt.ylabel("Тест 2")
plt.title('2 теста микрочипов. Логит с C=1')
plt.legend()

print("Доля правильных ответов классификатора на обучающей выборке:", round(logit.score(X_poly, y), 3))


C = 1e4
logit = LogisticRegression(C=C, random_state=17)
logit.fit(X_poly, y)

# plot_boundary(logit, X, grid_step=.005, poly_featurizer=poly)
#
# plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
# plt.xlabel("Тест 1")
# plt.ylabel("Тест 2")
# plt.title('2 теста микрочипов. Логит с C=10k')
# plt.legend()

print("Доля правильных ответов классификатора на обучающей выборке:", round(logit.score(X_poly, y), 3))
"""Еще увеличим  – до 10 тысяч. Теперь регуляризации явно недостаточно, и мы наблюдаем переобучение"""


"""Теперь найдем оптимальное (в данном примере) значение параметра регуляризации """
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

c_values = np.logspace(-2, 3, 500)

logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=1)
logit_searcher.fit(X_poly, y)

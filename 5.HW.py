import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV


warnings.filterwarnings('ignore')


def delete_nan(table):
    for col in table.columns:
        table[col] = table[col].fillna(table[col].median())
    return table


data = pd.read_csv('dataset/credit_scoring_sample.csv', sep=';')

# Посмотрим на распределение классов в зависимой переменной
# ax = data['SeriousDlqin2yrs'].hist(orientation='horizontal', color='red')
# ax.set_xlabel('number_of_observations')
# ax.set_ylabel("unique_value")
# ax.set_title("Target distribution")

# print(data['SeriousDlqin2yrs'].value_counts()/data.shape[0])

# Выберем названия всех признаков из таблицы, кроме прогнозируемого

independent_columns_names = [x for x in data.columns.values if x != 'SeriousDlqin2yrs']

table = delete_nan(data)

X = table[independent_columns_names]
y = table['SeriousDlqin2yrs']

"""Бутстреп"""
np.random.seed(0)
"""2. Сделайте интервальную оценку среднего возраста (age) для клиентов, 
которые просрочили выплату кредита, с 90% "уверенностью"."""


def stat_intervals(stat, alpha):
    # функция для интервальной оценки
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries


def get_bootstrap_samples(data, n_samples):
    # функция для генерации подвыборок с помощью бутстрэпа
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


arr = X[y.astype('bool')]['age'].values
mean_age = [np.mean(sample) for sample in get_bootstrap_samples(arr, arr.shape[0])]
# print(stat_intervals(mean_age, 0.1))


# Используем модуль LogisticRegression для построения логистической регрессии.
# Из-за несбалансированности классов  в таргете добавляем параметр балансировки.
# Используем также параметр random_state=5 для воспроизводимости результатов
lr = LogisticRegression(random_state=5, class_weight='balanced')

# Попробуем подобрать лучший коэффициент регуляризации для модели лог.регрессии.
# Остальные параметры оставляем по умолчанию.
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

"""3. GridSearch for C"""
grid_s = GridSearchCV(lr, parameters, scoring='roc_auc', cv=skf)
grid_s.fit(X, y)
print(grid_s.best_params_)
# 79%


"""4. Можно ли считать лучшую модель устойчивой? 
      Модель считаем устойчивой, если стандартное отклонение на валидации < 0.5%"""
grid_s.cv_results_['std_test_score'][1]


"""5. Определите самый важный признак """
lr.set_params(C=grid_s.best_params_['C'])
scal = StandardScaler()
lr.fit(scal.fit_transform(X), y)
# print(lr.coef_.argmax())

"""6. Посчитать долю влияния DebtRatio на предсказание"""
softmax = (np.exp(lr.coef_[0]) / np.sum(np.exp(lr.coef_[0])))[2]
# print(softmax)


"""7. Во сколько раз увеличатся шансы, что клиент не выплатит кредит, если увеличить возраст на 20 лет
при всех остальных неизменных значениях признаков"""

lr.fit(X, y)
print(np.exp(lr.coef_[0][0]*20))


"""Случайный лес"""
# Инициализируем случайный лес с 100 деревьями и сбалансированными классами
rf = RandomForestClassifier(n_estimators=100, random_state=42,
                            oob_score=True, class_weight='balanced')

# Будем искать лучшие параметры среди следующего набора
parameters = {'max_features': [1, 2, 4],
              'min_samples_leaf': [3, 5, 7, 9],
              'max_depth': [5, 10, 15]}

# Делаем опять же стрэтифайд k-fold валидацию. Инициализация которой должна у вас продолжать храниться в skf
rf_grid = GridSearchCV(rf, parameters, scoring='roc_auc', cv=skf)
rf_grid.fit(X, y)

# 83%
print(rf_grid.best_score_ - grid_s.best_score_)

"""Какой признак имеет самое слабое влияние в лучшей модели случайного леса?"""

print(X.columns[rf_grid.best_estimator_.feature_importances_.argmin()])


"""Бэггинг"""
"""11. Следующая задача обучить бэггинг классификатор .
В качестве базовых классификаторов возьмите 100 логистических регрессий и на этот раз используйте не GridSearchCV,
а RandomizedSearchCV. Так как перебирать все 54 варианта комбинаций долго, то поставьте максимальное число итераций 20
для RandomizedSearchCV."""

parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9],
              "base_estimator__C": [0.0001, 0.001, 0.01, 1, 10, 100]}

bagging_clf = BaggingClassifier(LogisticRegression(class_weight='balanced'), n_estimators=100, random_state=42)
grid_bag = RandomizedSearchCV(bagging_clf, parameters, scoring='roc_auc', cv=skf, random_state=1, n_iter=20)

grid_res = grid_bag.fit(X, y)
print(grid_res.best_score_)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


"""Пример визуального анализа данных"""
df = pd.read_csv('dataset/telecom_churn.csv')

# Посмотрим на распределение целевого класса - оттока клиентов
# df['Churn'].value_counts().plot(kind='bar', label='Churn')
# plt.legend()
# plt.title('Распределение оттока клиентов')
# plt.show()


"""Посмотрим на корреляции количественных признаков. По раскрашенной матрице корреляций видно, что такие признаки
 как Total day charge считаются по проговоренным минутам (Total day minutes).
  То есть 4 признака можно выкинуть, они не несут полезной информации."""

corr_matrix = df.drop(['State', 'International plan', 'Voice mail plan',
                      'Area code'], axis=1).corr()
# Удаляем не количественные признаки. Сorr - создает матрицу корреляции

# sns.heatmap(corr_matrix, cmap='magma_r')
# plt.show()

"""Матрица корреляции. На пересечении - линейный коэффициент корреляции (Пирсона) двух признаков.
 Характеризует существование ЛИНЕЙНОЙ связи между двумя величинами. Лежит в диапазоне от -1 до +1.
 1) Положительная корреляция (k > 0). При увеличении первой величины увеличивается вторая.
 2) Отрицательная корреляция (k < 0). При увеличении первой величины уменьшается вторая.
 3) Слабая корреляция (k ~~ 0). 
 """

"""Распределение количественных признаков"""
features = list(set(df.columns) - {'State', 'International plan', 'Voice mail plan',
                                   'Area code', 'Total day charge', 'Total eve charge',
                                   'Total night charge', 'Total intl charge', 'Churn'})

# df[features].hist(figsize=(20, 12))
# plt.show()
"""Большинство признаков распределены нормально"""

"""Распределение Customer service calls"""
# sns.countplot(x='Customer service calls', hue='Churn', data=df)  # Bar
# plt.show()


"""t-SNE
Основная идея: поиск нового представления данных, при котором сохраняется соседство"""


# Выкинем штаты, признак оттока,\
#  бинарные Yes/No признаки преобразуем в числа (pd.factorize - возвращает кортеж, результат факторизации и Uniques,
#  в нашем случае ['No', 'Yes'])

X = df.drop(['Churn', 'State'], axis=1)
X['International plan'] = pd.factorize(X['International plan'])[0]
X['Voice mail plan'] = pd.factorize(X['Voice mail plan'])[0]


"""Масштабируем выборку. Из каждого признака вычесть его среднее и поделить на стандартное отклонение
Этим занимается StandartScaler"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)

plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1])

"""Раскрасим полученное t-SNE представление данных по оттоку"""

plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=df['Churn'].map({0: 'blue', 1: 'orange'}))

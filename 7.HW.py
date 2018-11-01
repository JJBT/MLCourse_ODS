import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.family'] = 'DejaVU Sans'


X_train = np.loadtxt('dataset/samsung_HAR/samsung_train.txt')
y_train = np.loadtxt('dataset/samsung_HAR/samsung_train_labels.txt').astype('int')

X_test = np.loadtxt('dataset/samsung_HAR/samsung_test.txt')
y_test = np.loadtxt('dataset/samsung_HAR/samsung_test_labels.txt').astype('int')

X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])

n_classes = np.unique(y).size

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(random_state=17)
X_pca = pca.fit_transform(X_scaled)


"""Вопрос 1:
Какое минимальное число главных компонент нужно выделить, чтобы объяснить 90% дисперсии исходныхданных?"""
cum = np.cumsum(pca.explained_variance_ratio_)
# len(cum[cum < 0.9]) + 1
# 65 pc`s


"""Вопрос 2:
Сколько процентов дисперсии приходится на первую главную компоненту? Округлите до целых процентов."""
# print(round(pca.explained_variance_ratio_[0], 2))
# 51 %

# # Визуализация
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, cmap='viridis')
# plt.show()

"""Clustering"""
km = KMeans(n_clusters=n_classes, n_init=100, random_state=17)
pca.set_params(n_components=65)
X_pca = pca.fit_transform(X_scaled)

# X_pca = X_pca[:, 0:65]

km.fit(X_pca)

# # Визуализация кластеризация на первые две главные компоненты
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, s=20, cmap='viridis')
# plt.show()


tab = pd.crosstab(y, km.labels_, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице',
             'спуск по лестнице', 'сидение', 'стояние', 'лежание', 'все']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['все']


metrica = tab.iloc[0:6, 0:6].max(axis=1) / tab.drop('все', axis=0)['все']
# print(metrica.idxmax())


"""Видно, что kMeans не очень хорошо отличает только активности друг от друга. Используйте метод локтя, чтобы выбрать 
оптимальное количество кластеров. Параметры алгоритма и данные используем те же, что раньше, меняем только n_clusters."""

inertia = []

for k in range(1, n_classes + 1):
    kmeans = KMeans(n_clusters=k, random_state=17).fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 7), inertia, marker='o')
plt.xlabel('$k$')
plt.ylabel('$J(C_K)$')
plt.show()
# Оптимально - два кластера
#
"""АГЛОМЕРАТИВНАЯ"""
ag = AgglomerativeClustering(n_clusters=n_classes).fit(X_pca)
print(sklearn.metrics.adjusted_rand_score(y, ag.labels_))
print(sklearn.metrics.adjusted_rand_score(y, km.labels_))
# Справляются плохо


"""LinearSVC"""
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

svc = LinearSVC(random_state=17)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
best_svc = GridSearchCV(svc, svc_params, cv=3).fit(X_train_scaled, y_train)

# print(best_svc.best_params_)

y_predicted = best_svc.predict(X_test_scaled)

tab = pd.crosstab(y_test, y_predicted, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице', 'спуск по лестнице',
             'сидение', 'стояние', 'лежание', 'все']
tab.columns = tab.index

precision = tab.max() / tab.sum(axis=0)
recall = tab.max() / tab.sum(axis=1)


#
X_train_scaled_pca, X_test_scaled = pca.fit_transform(X_train_scaled), pca.fit_transform(X_test_scaled)
best_svc.fit(X_train_scaled_pca, y_train)

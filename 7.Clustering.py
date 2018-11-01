"""K-means"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


# Начнём с того, что насыпем на плоскость три кластера точек
X = np.zeros((150, 2))

np.random.seed(seed=42)
X[:50, 0] = np.random.normal(loc=0.0, scale=.3, size=50)
X[:50, 1] = np.random.normal(loc=0.0, scale=.3, size=50)

X[50:100, 0] = np.random.normal(loc=2.0, scale=.5, size=50)
X[50:100, 1] = np.random.normal(loc=-1.0, scale=.2, size=50)

X[100:150, 0] = np.random.normal(loc=-1.0, scale=.2, size=50)
X[100:150, 1] = np.random.normal(loc=2.0, scale=.5, size=50)

# plt.figure(figsize=(5, 5))
# plt.plot(X[:, 0], X[:, 1], 'bo')

# Располагаем центроиды
centroids = np.random.normal(loc=0, scale=1, size=6)
centroids = centroids.reshape((3, 2))

cent_history = list()
cent_history.append(centroids)

for i in range(3):
    # Расстояние от наблюдений до центроид. distances.shape = (150, 3) 150 точек, 3 центроида
    distances = cdist(X, centroids)
    # Смотрим, до какой центроиде каждой точке ближе всего
    labels = distances.argmin(axis=1)

    # Положим в каждую новую центроиду геометрический центр ее точек
    centroids = centroids.copy()
    centroids[0, :] = np.mean(X[labels == 0, :], axis=0)
    centroids[1, :] = np.mean(X[labels == 1, :], axis=0)
    centroids[2, :] = np.mean(X[labels == 2, :], axis=0)

    cent_history.append(centroids)

# # Визуализация
# for i in range(4):
#     distances = cdist(X, cent_history[i])
#     labels = distances.argmin(axis=1)
#
#     plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'bo', label='cluster #1')
#     plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'co', label='cluster #2')
#     plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'mo', label='cluster #3')
#     plt.plot(cent_history[i][:, 0], cent_history[i][:, 1], 'rX')
#     plt.legend(loc=0)
#     plt.title('Step {:}'.format(i + 1))
#     plt.show()


"""Выбор числа кластеров"""
inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 8), inertia, marker='s')
plt.show()
"""Оптимально - 3 кластера"""

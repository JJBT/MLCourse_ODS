import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


digits = datasets.load_digits()
X = digits.data
y = digits.target

#
# for i in range(10):
#     plt.imshow(X[i, :].reshape([8, 8]))
#     plt.show()
#

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)


# plt.figure(figsize=(20, 16))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolors='none', alpha=0.7,
#             cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.title('MNIST. PCA projection')
# plt.savefig('MNIST.png')
# plt.show()

"""PCA находит только линейные комбинации исходных признаков.
На практике выбирают столько главных компонент, чтобы оставить 90% дисперсии исходных данных."""

pca = PCA().fit(X)

# # График дисперсии выбранных компонент из исходных. Как видим 21 компонеты достаточно чтобы сохранить 90% дисперсии
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.axvline(21)
# plt.axhline(0.9)
# plt.show()


# logit = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)
#
# logit.fit(X_train, y_train)
# print(accuracy_score(y_test, logit.predict(X_test)))
#
# X_train1, X_test1, y_train1, y_test1 = train_test_split(PCA(n_components=21).fit_transform(X), y,
#                                                         test_size=0.3, stratify=y, shuffle=True)
# # rf.fit(X_train1, y_train1)
# logit2 = LogisticRegression()
# logit2.fit(X_train1, y_train1)
# print(accuracy_score(y_test, logit2.predict(X_test1)))
#

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from SGD import MySGDRegressor, MySGDClassifier

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.datasets import load_breast_cancer, load_files
from sklearn.feature_extraction.text import CountVectorizer

plt.style.use('ggplot')


data_demo = pd.read_csv('dataset/weights_heights.csv')
plt.scatter(data_demo['Weight'], data_demo['Height'])
plt.xlabel('Вес (фунты)')
plt.ylabel('Рост (дюймы)')
plt.show()

X, y = data_demo['Weight'].values, data_demo['Height'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape([X_train.shape[0], 1]))
X_valid_scaled = scaler.transform(X_valid.reshape([X_valid.shape[0], 1]))

sgd = MySGDRegressor(n_iter=10)
sgd.fit(X_train_scaled, y_train)

"""Loss-График"""
plt.plot(range(sgd.n_iter*X_train_scaled.shape[0]), sgd.mse_)
plt.show()

"""График весов"""
plt.plot([w[0] for w in sgd.weights_],
         [w[1] for w in sgd.weights_])
plt.show()

print(mean_squared_error(y_valid, sgd.predict(X_valid_scaled)))

linregr = LinearRegression()
linregr.fit(X_train_scaled, y_train)
print(mean_squared_error(y_valid, linregr.predict(X_valid_scaled)))


"""Логистическая регрессия и SGD"""
cancer = load_breast_cancer()

X, y = cancer.data, [-1 if i == 0 else 1 for i in cancer.target]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

sgdc = MySGDClassifier(C=1, eta=0.001, n_iter=3)
sgdc.fit(X_train_scaled, y_train)

"""Loss-График"""
plt.plot(sgdc.loss_)
plt.show()

sgdc = MySGDClassifier(C=1000, eta=0.001, n_iter=10)
sgdc.fit(X_train_scaled, y_train)

"""Loss-График"""
plt.plot(sgdc.loss_)
plt.show()

"""Какой признак сильнее остальных влияет на вероятность того, что опухоль доброкачественна"""
print(cancer.feature_names[np.argmax(sgdc.w_[1:])])
print(cancer.feature_names[np.argmin(sgdc.w_[1:])])

best_w = sgdc.w_
imp = pd.DataFrame({'coef': best_w,
             'feat': ['intercept'] + list(cancer.feature_names)}).sort_values(by='coef')


pred = sgdc.predict_proba(X_valid_scaled)
print('logloss and roc-auc for MySGD', log_loss(y_valid, pred), roc_auc_score(y_valid, pred))

logit = LogisticRegression(random_state=17)
logit.fit(X_train_scaled, y_train)

log_pred = logit.predict_proba(X_valid_scaled)[:, 1]
print('logloss and roc-auc for LOGIT', log_loss(y_valid, log_pred), roc_auc_score(y_valid, log_pred))


"""3. Логистическая регрессия и SGDClassifier в задаче классификации отзывов к фильмам"""
reviews_train = load_files('dataset/imdb_reviews/imdb_reviews/train')
text_train, y_train = reviews_train.data, reviews_train.target

reviews_test = load_files('dataset/imdb_reviews/imdb_reviews/test')
text_test, y_test = reviews_test.data, reviews_test.target

cv = CountVectorizer(ngram_range=(1, 2))
X_train = cv.fit_transform(text_train)
X_test = cv.transform(text_test)

logit = LogisticRegression(random_state=17)
logit.fit(X_train, y_train)

print('roc-auc logit', roc_auc_score(y_test, logit.predict_proba(X_test)[:, 1]))

sgd_logit = SGDClassifier(loss='log', random_state=17, n_iter=100)

sgd_logit.fit(X_train, y_train)
print('roc-auc sgd-logit', roc_auc_score(y_test, sgd_logit.predict_proba(X_test)[:, 1]))

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, log_loss


class MySGDRegressor(BaseEstimator):
    def __init__(self, eta=0.001, n_iter=10):
        self.n_iter = n_iter
        self.eta = eta
        self.mse_ = np.array([])
        self.weights_ = np.array([])
        self.w = np.array([])
        self.w_ = np.array([])

    def fit(self, X, y):
        self.X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.zeros(self.X_.shape[1])

        for i in range(self.X_.shape[0] * self.n_iter):
            for j in range(self.w.shape[0]):
                self.w[j] = self.w[j] + self.eta*(y[i % len(y)] -
                            np.dot(self.X_[i % self.X_.shape[0]], self.w)) * self.X_[i % self.X_.shape[0], j]

            self.mse_ = np.append(self.mse_, mean_squared_error(y, np.dot(self.X_, self.w)))
            self.weights_ = np.append(self.weights_, self.w)

        self.weights_ = self.weights_.reshape(-1, self.X_.shape[1])
        self.w_ = self.weights_[np.argmin(self.mse_)]
        return self

    def predict(self, X):
        return np.dot(np.hstack([np.ones((X.shape[0], 1)), X]), self.w_)


""""""


def sigma(z):
    z = z.flatten()
    z[z > 100] = 100
    z[z < -100] = -100
    return 1. / (1 + np.exp(-z))


class MySGDClassifier(BaseEstimator):
    def __init__(self, C, eta=0.001, n_iter=10):
        self.C = C
        self.n_iter = n_iter
        self.eta = eta
        self.loss_ = np.array([])
        self.weights_ = np.array([])
        self.w = np.array([])
        self.w_ = np.array([])

    def fit(self, X, y):
        self.X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.ones(self.X_.shape[1])
        self.w_ = np.zeros(self.X_.shape[1])

        for i in range(self.n_iter * self.X_.shape[0]):
            for j in range(self.w.shape[0]):
                self.w[j] = self.w[j] + self.eta*(self.C * y[i % len(y)] * self.X_[i % self.X_.shape[0], j] *
                            sigma(-y[i % len(y)] * np.dot(self.X_[i % self.X_.shape[0]], self.w)) - int(j != 0)*self.w[j])

            self.loss_ = np.append(self.loss_, log_loss(y, sigma(np.dot(self.X_, self.w))))
            self.weights_ = np.append(self.weights_, self.w)

        self.weights_ = self.weights_.reshape(-1, self.X_.shape[1])
        self.w_ = self.weights_[np.argmin(self.loss_)]
        return self

    def predict_proba(self, X):
        return sigma(np.dot(np.hstack([np.ones((X.shape[0], 1)), X]), self.w_))

    def predict(self, X):
        self.labels = np.sign(self.predict_proba(X) - 0.5)
        self.labels[np.where(self.labels == 0)] = 1
        return self.labels



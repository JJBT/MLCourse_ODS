import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import random


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


def make_data(N, err=1.0, rseed=1):
    # Создаем случайные выборки данных
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y



X = np.array([[1, 2, 3],
              [3, 4, 5]])


def _find_splits(X):
    """Find all possible split values."""
    split_values = set()

    # Get unique values in a sorted order
    x_unique = list(np.unique(X))
    print(x_unique)
    for i in range(1, len(x_unique)):
        # Find a point between two values
        average = (x_unique[i - 1] + x_unique[i]) / 2.0
        split_values.add(average)

    return list(split_values)


print(_find_splits(X))


subset = random.sample(list(range(0, X.shape[1])), 3)

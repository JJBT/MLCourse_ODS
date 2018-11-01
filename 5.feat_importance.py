"""Оценка важности признаков"""

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
# russian headers
from matplotlib import rc
font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


hostel_data = pd.read_csv('dataset/hostel_factors.csv')
features = {"f1": u"Персонал",
            "f2": u"Бронирование хостела ",
            "f3": u"Заезд в хостел и выезд из хостела",
            "f4": u"Состояние комнаты",
            "f5": u"Состояние общей кухни",
            "f6": u"Состояние общего пространства",
            "f7": u"Дополнительные услуги",
            "f8": u"Общие условия и удобства",
            "f9": u"Цена/качество",
            "f10": u"ССЦ"}

forest = RandomForestRegressor(n_estimators=1000, max_features=10, random_state=0)

forest.fit(hostel_data.drop(['hostel', 'rating'], axis=1), hostel_data['rating'])
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

num_to_plot = 10

feature_indices = [ind + 1 for ind in indices[:num_to_plot]]

"""Visualization"""
plt.bar(range(num_to_plot), importances[indices[:num_to_plot]])
plt.xticks(range(num_to_plot), feature_indices)
plt.legend()
plt.show()

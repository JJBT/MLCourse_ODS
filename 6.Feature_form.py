import pandas as pd
import reverse_geocoder as revgc
import user_agents
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


"""Feature Extraction. Извлечение признаков"""
"""Работа с текстом в файле Bag_of_words.py"""


"""Геоданные"""
# revgc.search(('40.74482', '-73.94875'))


"""Веб. Юзер-Агент"""
ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/' \
     '...: 56.0.2924.76 Safari/537.36'
ua = user_agents.parse(ua)

print(ua.is_mobile)
print(ua.os.family)


"""Feature selection"""

"""Отсекать признаки, дисперсия которых ниже определенной границы."""
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

X_data_generated, y_data_generated = make_classification()
print(X_data_generated.shape)
# (100, 20)
print(VarianceThreshold(0.9).fit_transform(X_data_generated).shape)
# (100, 14)

"""Отбор с использованием модели"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


pipe = make_pipeline(SelectFromModel(estimator=RandomForestClassifier()),
                     LogisticRegression())

lr = LogisticRegression()
rf = RandomForestClassifier()

print(cross_val_score(lr, X_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(rf, X_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(pipe, X_data_generated, y_data_generated, scoring='neg_log_loss').mean())


"""Перебор"""
selector = SFS(LogisticRegression(), scoring='neg_log_loss', verbose=2, k_features=3,
               forward=False, n_jobs=-1)
selector.fit(X_data_generated, y_data_generated)



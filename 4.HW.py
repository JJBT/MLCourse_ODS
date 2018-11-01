"""Прогнозирование популярности статей на Хабре с помощью линейных моделей"""

import numpy as np
import pandas as pd
import scipy

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


train_df = pd.read_csv('dataset/howpop_train.csv')
test_df = pd.read_csv('dataset/howpop_test.csv')


"""Вопрос 1. Есть ли в train_df признаки, корреляция между которыми больше 0.9?"""
c = train_df.corr()
c = c.applymap(lambda x: x > 0.9)
c = c.apply(lambda s: sum(s) > 1)
print(any(c))

"""Вопрос 2. В каком году было больше всего публикаций? (Рассматриваем train_df)"""
train_df['published'] = pd.to_datetime(train_df['published'])
train_df['years'] = train_df['published'].apply(lambda x: x.year)
print(train_df['years'].value_counts().head(1))


"""Разбиение на train/valid"""
features = ['author', 'flow', 'domain', 'title']
train_size = int(0.7 * train_df.shape[0])

X, y = train_df.loc[:, features], train_df['favs_lognorm']
X_test = test_df.loc[:, features]

X_train, X_valid = X.iloc[:train_size, :], X.iloc[train_size:, :]
y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]

tfidf = TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 3))
X_train_title = tfidf.fit_transform(X_train['title'])
X_valid_title = tfidf.transform(X_valid['title'])
X_test_title = tfidf.transform(X_test['title'])


"""Вопрос 3. Какой размер у полученного словаря?"""
print(len(tfidf.vocabulary_))

"""Вопрос 4. Какой индекс у слова 'python'?"""
print(tfidf.vocabulary_['python'])


vectorizer_title_ch = TfidfVectorizer(analyzer='char')

X_train_title_ch = vectorizer_title_ch.fit_transform(X_train['title'])
X_valid_title_ch = vectorizer_title_ch.transform(X_valid['title'])
X_test_title_ch = vectorizer_title_ch.transform(X_test['title'])

"""Вопрос 5. Какой размер у полученного словаря?"""
print(len(vectorizer_title_ch.vocabulary_))


"""Работа с категориальными признаками"""

feats = ['author', 'flow', 'domain']
dict_vect = DictVectorizer()
dict_vect_matrix = dict_vect.fit_transform(X_train[feats][:5].fillna('-').T.to_dict().values())
print(dict_vect.fit_transform(X_train[feats][:5].fillna('-').T.to_dict()))
print(type(dict_vect.fit_transform(X_train[feats][:5].fillna('-').T.to_dict())))
print(dict_vect.fit_transform(X_train[feats][:5].fillna('-').T.to_dict()).values())
print(type(dict_vect.fit_transform(X_train[feats][:5].fillna('-').T.to_dict()).values()))

vectorizer_feats = DictVectorizer()

X_train_feats = vectorizer_feats.fit_transform(X_train[feats].fillna('-').T.to_dict().values())
X_valid_feats = vectorizer_feats.transform(X_valid[feats].fillna('-').T.to_dict().values())
X_test_feats = vectorizer_feats.transform(X_test[feats].fillna('-').T.to_dict().values())


X_train_new = scipy.sparse.hstack([X_train_title, X_train_feats, X_train_title_ch])
X_valid_new = scipy.sparse.hstack([X_valid_title, X_valid_feats, X_valid_title_ch])
X_test_new = scipy.sparse.hstack([X_test_title, X_test_feats, X_test_title_ch])

r1 = Ridge(alpha=0.1, random_state=1).fit(X_train_new, y_train)
r2 = Ridge(alpha=1, random_state=1).fit(X_train_new, y_train)

train_preds1 = r1.predict(X_train_new)
valid_preds1 = r1.predict(X_valid_new)

print('Ошибка на трейне', mean_squared_error(y_train, train_preds1))
print('Ошибка на тесте', mean_squared_error(y_valid, valid_preds1))


train_preds2 = r2.predict(X_train_new)
valid_preds2 = r2.predict(X_valid_new)

print('Ошибка на трейне', mean_squared_error(y_train, train_preds2))
print('Ошибка на тесте', mean_squared_error(y_valid, valid_preds2))


"""Baseline"""
model = Ridge()

model.fit(scipy.sparse.vstack([X_train_new, X_valid_new]), y)
test_preds = model.predict(X_test_new)

sample_submission = pd.read_csv('dataset/habr_sample_submission.csv', index_col='url')

ridge_submission = sample_submission.copy()
ridge_submission['favs_lognorm'] = test_preds

ridge_submission.to_csv('ridge_baseline.csv')

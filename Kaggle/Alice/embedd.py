"""Catch me if you can"""


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


train_df = pd.read_csv('Kaggle/Alice/train_sessions.csv', index_col='session_id')
test_df = pd.read_csv('Kaggle/Alice/test_sessions.csv', index_col='session_id')

times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

train_df = train_df.sort_values(by='time1')

sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int').astype('str')
test_df[sites] = test_df[sites].fillna(0).astype('int').astype('str')

train_df['list'] = train_df['site1']
test_df['list'] = test_df['site1']

for s in sites[1:]:
    train_df['list'] = train_df['list'] + ',' + train_df[s]
    test_df['list'] = test_df['list'] + ',' + test_df[s]

train_df['list_w'] = train_df['list'].apply(lambda x: x.split(','))
test_df['list_w'] = test_df['list'].apply(lambda x: x.split(','))


from gensim.models import word2vec

test_df['target'] = -1
data = pd.concat([train_df, test_df], axis=0)

model = word2vec.Word2Vec(data['list_w'], size=300, window=3, workers=4)

w2v = dict(zip(model.wv.index2word, model.wv.syn0))


class MeanVectorizer:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(w2v.values())))

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


data_mean = MeanVectorizer(w2v).fit(train_df['list_w']).transform(train_df['list_w'])


from sklearn.model_selection import train_test_split

y = train_df['target']

X_train, X_val, y_train, y_val = train_test_split(data_mean, y, test_size=0.2)


def get_auc_lr_valid(X, y, C=1, seed=7, ratio=0.8):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-ratio)

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=C, random_state=seed, n_jobs=-1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_val)[:, 1]

    score = roc_auc_score(y_val, y_pred)

    return score


print(get_auc_lr_valid(data_mean, y, ratio=0.7))

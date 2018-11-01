"""Catch me if you can"""

import numpy as np
import pandas as pd
import pickle
# from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit


train_df = pd.read_csv('dataset/Kaggle/Alice/train_sessions.csv')
test_df = pd.read_csv('dataset/Kaggle/Alice/test_sessions.csv')


times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

train_df.sort_values(by='time1', inplace=True)


sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# Словарь сайтов
with open(r"dataset/Kaggle/Alice/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)


sites_dict_df = pd.DataFrame(list(site_dict.keys()),
                             index=list(site_dict.values()),
                             columns=['site'])

y_train = train_df['target']
# Объединенная таблица данных
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Индекс, по которому отделяется обучающая выборка от тестовой
idx_split = train_df.shape[0]

full_sites = full_df[sites]
# последовательность с индексами
sites_flatten = full_sites.values.flatten()

# искомая разреженная матрица ?????????????
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

X_train_sparse = full_sites_sparse[:idx_split]
X_test_sparse = full_sites_sparse[idx_split:]


full_times = full_df[times]


def get_auc_lr_valid(X, y, C=1.0, ratio=0.9, seed=17):
    train_len = int(ratio * X.shape[0])
    X_train = X[:train_len, :]
    X_valid = X[train_len:, :]
    y_train = y[:train_len]
    y_valid = y[train_len:]

    logit = LogisticRegression(C=C, random_state=seed, n_jobs=-1)
    logit.fit(X_train, y_train)

    valid_pred = logit.predict_proba(X_valid)[:, 1]

    return roc_auc_score(y_valid, valid_pred)


# print(get_auc_lr_valid(X_train_sparse, y_train))


def write_to_submission_file(predicted_labels, out_files, target='target',
                             index_label='session_id'):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1, predicted_labels.shape[0] + 1),
                                columns=['target'])
    predicted_df.to_csv(out_files, index_label=index_label)


logit = LogisticRegression(n_jobs=-1)
# logit.fit(X_train_sparse, y_train)
# test_preds = logit.predict_proba(X_test_sparse)[:, 1]
#

"""Улучшение модели, построение новых признаков"""
new_feat_train = pd.DataFrame(index=train_df.index)
new_feat_test = pd.DataFrame(index=test_df.index)
new_feat_train['year_month'] = train_df['time1']\
    .apply(lambda ts: 100 * ts.year + ts.month)
new_feat_test['year_month'] = test_df['time1']\
    .apply(lambda ts: 100 * ts.year + ts.month)

scaler = StandardScaler()
scaler.fit(new_feat_train['year_month'].values.reshape(-1, 1))

new_feat_train['year_month_sc'] = scaler.transform(new_feat_train['year_month'].values.reshape(-1, 1))
new_feat_test['year_month_sc'] = scaler.transform(new_feat_test['year_month'].values.reshape(-1, 1))

X_train_sparse_new = csr_matrix(hstack([X_train_sparse, new_feat_train['year_month_sc'].values.reshape(-1, 1)]))

# print(get_auc_lr_valid(X_train_sparse_new, y_train))

new_feat_train['start_hour'] = train_df['time1']\
    .apply(lambda ts: ts.hour)
new_feat_test['start_hour'] = test_df['time1']\
    .apply(lambda ts: ts.hour)

new_feat_train['start_month'] = train_df['time1']\
    .apply(lambda ts: ts.month)
new_feat_test['start_month'] = test_df['time1']\
    .apply(lambda ts: ts.month)

new_feat_train['morning'] = new_feat_train['start_hour']\
    .apply(lambda ts: (ts >= 7) & (ts <= 11)).astype('int')
new_feat_test['morning'] = new_feat_test['start_hour']\
    .apply(lambda ts: (ts >= 7) & (ts <= 11)).astype('int')

new_feat_train['day'] = new_feat_train['start_hour']\
    .apply(lambda ts: (ts >= 12) & (ts <= 18)).astype('int')
new_feat_test['day'] = new_feat_test['start_hour']\
    .apply(lambda ts: (ts >= 12) & (ts <= 18)).astype('int')

new_feat_train['evening'] = new_feat_train['start_hour']\
    .apply(lambda ts: (ts >= 19) & (ts <= 23)).astype('int')
new_feat_test['evening'] = new_feat_test['start_hour']\
    .apply(lambda ts: (ts >= 19) & (ts <= 23)).astype('int')

new_feat_train['night'] = new_feat_train['start_hour']\
    .apply(lambda ts: (ts >= 0) & (ts <= 6)).astype('int')
new_feat_test['night'] = new_feat_test['start_hour']\
    .apply(lambda ts: (ts >= 0) & (ts <= 6)).astype('int')


new_feat_train['total_sites'] = full_sites[:idx_split].apply(lambda x: len(x.nonzero()[0]), axis=1)
new_feat_test['total_sites'] = full_sites[idx_split:].apply(lambda x: len(x.nonzero()[0]), axis=1)

time_delt = full_times.apply(lambda s: s.max() - s[0], axis=1)

time_delt_sec = time_delt.apply(lambda t: t.seconds)

scaler.fit(time_delt_sec.values.reshape(-1, 1))
time_delt_sec_scaled = scaler.transform(time_delt_sec.values.reshape(-1, 1))

new_feat_train['total_sec'] = time_delt_sec_scaled[:idx_split]
new_feat_test['total_sec'] = time_delt_sec_scaled[idx_split:]


# print(get_auc_lr_valid(csr_matrix(hstack([X_train_sparse, new_feat_train[['start_month', 'start_hour']].values.reshape(-1, 2)])), y_train))
# print(get_auc_lr_valid(csr_matrix(hstack([X_train_sparse, new_feat_train[['start_month', 'morning']].values.reshape(-1, 2)])), y_train))
# print(get_auc_lr_valid(csr_matrix(hstack([X_train_sparse, new_feat_train[['start_month', 'start_hour', 'morning']].values.reshape(-1, 3)])), y_train))

mm_train = csr_matrix(hstack([X_train_sparse, new_feat_train[['start_month', 'start_hour', 'morning', 'day', 'evening', 'night', 'total_sites', 'total_sec']].values.reshape(-1, 8)]))
mm_test = csr_matrix(hstack([X_test_sparse, new_feat_test[['start_month', 'start_hour', 'morning', 'day', 'evening', 'night', 'total_sites', 'total_sec']].values.reshape(-1, 8)]))


# logit.fit(mm_train, y_train)
# test_preds = logit.predict_proba(mm_test)[:, 1]


"""Подбор коефициента регуляризации"""
C = np.logspace(-3, 1, 10)
time_split = TimeSeriesSplit(n_splits=10)
logitCV = LogisticRegressionCV(Cs=C, cv=time_split, scoring='roc_auc')
logitCV.fit(mm_train, y_train)


# print(get_auc_lr_valid(mm_train, y_train, C=logitCV.C_[0]))





import re

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


newsgroups = fetch_20newsgroups('dataset/news_data')
target = newsgroups['target_names'][newsgroups['target'][0]]


def to_vw_format(document, label=None):
    return str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'


all_documents = newsgroups['data']
all_targets = [1 if newsgroups['target_names'][target] == 'rec.autos' else -1 for target in newsgroups['target']]

train_documents, test_documents, train_labels, test_labels = \
    train_test_split(all_documents, all_targets, random_state=17)


with open('dataset/news_data/20news_train.vw', 'w') as vw_train_data:
    for text, target in zip(train_documents, train_labels):
        try:
            vw_train_data.write(to_vw_format(text, target))
        except UnicodeEncodeError:
            pass

with open('dataset/news_data/20news_test.vw', 'w') as vw_test_data:
    for text in test_documents:
        try:
            vw_test_data.write(to_vw_format(text))
        except UnicodeEncodeError:
            pass

"""!vw -d dataset/news_data/20news_train.vw \
  --loss_function hinge -f dataset/news_data/20news_model.vw"""

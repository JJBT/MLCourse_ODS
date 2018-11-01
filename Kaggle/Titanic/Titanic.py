"""Titanic"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


"""Подготовка данных"""

train_df = pd.read_csv('dataset/Kaggle/Titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('dataset/Kaggle/Titanic/test.csv', index_col='PassengerId')

y = train_df['Survived']
train_df.drop('Survived', inplace=True, axis=1)


train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})
test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})

train_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
test_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

train_df = pd.concat([train_df, pd.get_dummies(train_df['Pclass'], prefix='Pclass')], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['Pclass'], prefix='Pclass')], axis=1)

train_df.drop('Pclass', axis=1, inplace=True)
test_df.drop('Pclass', axis=1, inplace=True)

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Age'] = train_df['Age'].astype('int64')

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Age'] = test_df['Age'].astype('int64')

train_df[['Pclass_1', 'Pclass_2', 'Pclass_3']] = train_df[['Pclass_1', 'Pclass_2', 'Pclass_3']].astype('int64')
test_df[['Pclass_1', 'Pclass_2', 'Pclass_3']] = test_df[['Pclass_1', 'Pclass_2', 'Pclass_3']].astype('int64')

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())


X_train, X_valid, y_train, y_valid = train_test_split(train_df, y, test_size=0.3, random_state=17)

#

"""Модель"""
logit = LogisticRegression(random_state=17)

logit.fit(X_train, y_train)
print(accuracy_score(y_valid, logit.predict(X_valid)))

c = np.logspace(-3, 2, 15)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=17)

logit_searcher = LogisticRegressionCV(Cs=c, cv=skf, scoring='accuracy', random_state=17)
logit_searcher.fit(train_df, y)


def write_to_submission_file(predicted_labels, out_files, target='target',
                             index_label='PassengerId'):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(892, predicted_labels.shape[0] + 892),
                                columns=['Survived'])
    predicted_df.to_csv(out_files, index_label=index_label)


test_preds = logit_searcher.predict(test_df)

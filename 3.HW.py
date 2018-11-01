from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
plt.rcParams['figure.figsize'] = (10, 8)


# Создание датафрейма с dummy variables
def create_df(dic, feature_list):
    out = pd.DataFrame(dic)
    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis=1)
    out.drop(feature_list, axis=1, inplace=True)
    return out


# Некоторые значения признаков есть в тесте, но нет в трейне и наоборот
def intersect_features(train, test):
    common_feat = list(set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]


features = ['Внешность', 'Алкоголь_в_напитке',
            'Уровень_красноречия', 'Потраченные_деньги']

# Обучающая выборка
df_train = dict()
df_train['Внешность'] = ['приятная', 'приятная', 'приятная', 'отталкивающая',
                         'отталкивающая', 'отталкивающая', 'приятная']
df_train['Алкоголь_в_напитке'] = ['да', 'да', 'нет', 'нет', 'да', 'да', 'да']
df_train['Уровень_красноречия'] = ['высокий', 'низкий', 'средний', 'средний', 'низкий',
                                   'высокий', 'средний']
df_train['Потраченные_деньги'] = ['много', 'мало', 'много', 'мало', 'много',
                                  'много', 'много']
df_train['Поедет'] = LabelEncoder().fit_transform(['+', '-', '+', '-', '-', '+', '+'])

df_train = create_df(df_train, features)

# Тестовая выборка
df_test = dict()
df_test['Внешность'] = ['приятная', 'приятная', 'отталкивающая']
df_test['Алкоголь_в_напитке'] = ['нет', 'да', 'да']
df_test['Уровень_красноречия'] = ['средний', 'высокий', 'средний']
df_test['Потраченные_деньги'] = ['много', 'мало', 'много']
df_test = create_df(df_test, features)

# Некоторые значения признаков есть в тесте, но нет в трейне и наоборот
y = df_train['Поедет']
df_train, df_test = intersect_features(train=df_train, test=df_test)


def entropy(feature):
    feature = pd.Series(feature)
    arr = list(feature.value_counts())
    entr = 0
    for i in range(len(arr)):
        entr += arr[i]/len(feature) * np.log2(arr[i]/len(feature))
    return -1*entr


def ig(Y, y_left, y_right):
    H = entropy(Y)
    elems_left = len(y_left) / len(Y)
    elems_right = len(y_right) / len(Y)
    return H - elems_left * entropy(y_left) - elems_right * entropy(y_right)


def best_feature_to_split(X, y):
    """Выводит прирост информации при разбиении по каждому признаку"""
    out = []
    for i in X.columns:
        out.append(ig(y, y[X[i] == 0], y[X[i] == 1]))
    return out


def btree(X, y):
    """Дерево решений"""
    clf = best_feature_to_split(X, y)
    param = clf.index(max(clf))
    ly = y[X.ix[:, param] == 0]
    ry = y[X.ix[:, param] == 1]
    print(ly, ry, sep='\n')
    print('Column_' + str(param) + ' N/Y?')
    print('Entropy: ', entropy(ly), entropy(ry))
    print('N count:', ly.count(), '/', 'Y count:', ry.count())
    if entropy(ly) != 0:
        left = X[X.ix[:, param] == 0]
        btree(left, ly)
    if entropy(ry) != 0:
        right = X[X.ix[:, param] == 1]
        btree(right, ry)


df_train['Поедет'] = y
df_train.sort_values(by='Внешность_приятная', inplace=True, ascending=False)
"""Вопрос 1. Какова энтропия начальной системы (S0)?
 Под состояниями системы понимаем значения признака "Поедет" – 0 или 1 (то есть всего 2 состояния)."""
entropy(df_train['Поедет'])

"""Вопрос 2. Рассмотрим разбиение обучающей выборки по признаку "Внешность_приятная". 
Какова энтропия S1 левой группы, тех, у кого внешность приятная, и правой группы – S2?
 Каков прирост информации при данном разбиении (IG)?"""

# print(entropy(df_train[df_train['Внешность_приятная'] == 1]['Поедет']))
ig1 = ig(df_train['Поедет'], df_train[df_train['Внешность_приятная'] == 1]['Поедет'], df_train[df_train['Внешность_приятная'] == 0]['Поедет'])
# print(entropy(df_train[df_train['Внешность_отталкивающая'] == 1]['Поедет']))


tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
df_train.drop(['Поедет'], axis=1, inplace=True)
tree.fit(df_train, y)


balls = [1 for _ in range(9)] + [0 for _ in range(11)]

# две группы
balls_left = [1 for _ in range(8)] + [0 for _ in range(5)] # 8 синих и 5 желтых
balls_right = [1 for _ in range(1)] + [0 for _ in range(6)] # 1 синий и 6 желтых

# print(entropy(balls))  # 9 синих и 11 желтых
# print(entropy(balls_left))  # 8 синих и 5 желтых
# print(entropy(balls_right))  # 1 синий и 6 желтых
# print(entropy([1, 2, 3, 4, 5, 6]))  # энтропия игральной кости с несмещенным центром тяжести
# print(ig(balls, balls_left, balls_right))

btree(df_train, y)

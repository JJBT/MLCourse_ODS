"""Урок 1 - Первичный анализ данных с Pandas"""

import pandas as pd
import numpy as np


"""Series - одномерный индексированный массив
   DataFrame - двухмерная структура данных, столбец - данные одного типа"""

df = pd.read_csv("dataset/telecom_churn.csv")  # чтение данных

print(df.head())  # первые 5 строк
print(df.shape)  # кортеж размерности

print(df.columns)  # название столбцов

print(df.info())  # информация по датафрейму и всем признакам

df['Churn'] = df['Churn'].astype('int64')  # изменить тип колонки

print(df.describe())  # основные статистические хар-ки по каждому числовому признаку
#  (среднее, стандартное отклонение, диапазон, медиана, квартили)

print(df['Churn'].value_counts())  # для категориальных (object) и булевых. Возвращает кол-во уникальных значений
print(df['Area code'].value_counts(normalize=True))  # normalize=True - относительные частоты

df.sort_values(by='Total day charge', ascending=False)

print(df['Churn'].mean())  # среднее значение


"""Логическая индексация  -  df[ P( df['Name'] ) ], где P — это некоторое логическое условие, 
проверяемое для каждого элемента столбца Name. Итогом такой индексации является DataFrame, состоящий только из строк, 
удовлетворяющих условию P по столбцу Name."""

"""Разные примеры индексации - по номеру, и по названию"""

print(df.iloc[0:5, 0:3])  # слайсится как обычно

print(df.loc[0:5, 'State':'Area code'])  # учитывается и начало и конец


print(df.apply(np.max))  # применение функции к каждому столбцу - возвращается одна строка
print(df.apply(all, axis=1))  # применение функции к каждой строке - возвращается один столбец


"""Применение функции к каждой ячейке столбца"""

df['International plan'] = df['International plan'].map({'No': False, 'Yes': True})
# map можно использовать для замены значений в колонке, передав ему словарь вида {old_value: new_value}

df = df.replace({'Voice mail plan': {'No': False, 'Yes': True}})  # Аналогичная операция


"""Группировка данных  
df.groupby(by=grouping_columns)[columns_to_show].function() 
1. К датафрейму применяется метод groupby, который разделяет данные по grouping_columns – признаку или набору признаков.
2. Выбираем нужные нам столбцы (columns_to_show).
3. К полученным группам применяется функция или несколько функций."""

# группирование данных в зависимости от значения признака Churn и вывод статистик по трем столбцам в каждой группе
df.groupby(by=['Churn'])[['Total day minutes', 'Total eve minutes', 'Total night minutes']].describe()

# то же самое, передав в agg список функций
df.groupby(['Churn'])[['Total day minutes', 'Total eve minutes', 'Total night minutes']].agg([np.mean, np.min, np.max])


print(pd.crosstab(df['Churn'], df['International plan']))  # сопряженная таблицы

print(df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
                     ['Area code'], aggfunc='max'))  # сводная таблица


"""Преобразование датафреймов"""

# Вставка
total_calls = df['Total day calls'] + df['Total eve calls'] + \
              df['Total night calls'] + df['Total intl calls']
df.insert(loc=len(df.columns) - 1, column='Total calls', value=total_calls)  # loc - номер столбца,
# после которого нужно вставить данный Series

# Удаление
df = df.drop(['Total calls'], axis=1)  # axis=1 для столбцов, axis=0 для строк
df = df.drop([1, 2])  # Удаление строк

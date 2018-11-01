"""Урок 2. Визуальный анализ"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Изменение дефолтного размера графика
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

df = pd.read_csv('dataset/video_games_sales.csv')


# dropna удаляет строчки в которых есть пропуск
df = df.dropna()

# Некоторые признаки, которые pandas считал как object, явно приведем к численным типам
df['User_Score'] = df['User_Score'].astype('float64')
df['Year_of_Release'] = df['Year_of_Release'].astype('int64')
df['User_Count'] = df['User_Count'].astype('int64')
df['Critic_Count'] = df['Critic_Count'].astype('int64')


useful_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre',
               'Global_Sales', 'Critic_Score', 'Critic_Count',
               'User_Score', 'User_Count', 'Rating']


# Простой пример визуализировать данные (plot)
# График продаж в различных странах в зависимости от года
# Фильтруем нужные столбцы, посчитаем суммарные продажи по годам
sales_df = df[[x for x in df.columns if 'Sales' in x] + ['Year_of_Release']]
new_df = sales_df.groupby('Year_of_Release').sum()
new_df.plot()


"""Seaborn. Построен на matplotlib"""

"""pair plot (scatter plot). Поможет посмотреть как связаны между собой различные признаки"""
cols = ['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
# sns_plot = sns.pairplot(df[cols])
# sns_plot.savefig('pairplot.png')

"""Распределение"""
# sns.distplot(df['Critic_Score'])


"""Взаимосвязь двух численных признаков. Гибрид scatter, histogram"""
# sns.jointplot(df['Critic_Score'], df['User_Score'])


"""Heat map позволяет посмотреть на распределение какого-то численного признака по двум категориальным. 
Визуализируем суммарные продажи игр по жанрам и игровым платформам."""
platform_genre_sales = df.pivot_table(
    index='Platform',
    columns='Genre',
    values='Global_Sales',
    aggfunc=sum).fillna(0).applymap(float)
"""fillna - для заполнения пропусков. applymap - применяет ф-ию к каждой ячейке в таблице"""

sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=.5)

plt.show()

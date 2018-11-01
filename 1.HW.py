"""Домашнее задание к уроку 1"""

import pandas as pd

data = pd.read_csv('dataset/adult.data.csv')

"""1. Сколько мужчин и женщин (признак sex) представлено в этом датасете"""
women = data[data['sex'] == 'Female']
men = data[data['sex'] == 'Male']

print('Female', len(women))
print('Male', len(men))


"""2. Каков средний возраст (признак age) женщин?"""
print(women['age'].mean())


"""3. Какова доля граждан Германии (признак native-country)?"""
print(data['native-country'].value_counts(normalize=True)['Germany'])


"""4-5. Каковы средние значения и среднеквадратичные отклонения возраста тех, кто получает более 50K в год
 и тех, кто получает менее 50K в год? (признак salary) """
print(data[data['salary'] == '>50K']['age'].describe())
print(data[data['salary'] == '<=50K']['age'].describe())


"""6. Правда ли, что люди, которые получают больше 50k, имеют как минимум высшее образование? 
(признак education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters или Doctorate)"""
tab = pd.crosstab(data['salary'], data['education'], normalize=True)
print(tab.loc['>50K'])
print((tab.loc['>50K', 'Assoc-acdm'] + tab.loc['>50K', 'Assoc-voc'] + tab.loc['>50K', 'Bachelors']
      + tab.loc['>50K', 'Doctorate'] + tab.loc['>50K', 'Masters'] + tab.loc['>50K', 'Prof-school']) == 1)


"""7. Выведите статистику возраста для каждой расы (признак race) и каждого пола. Используйте groupby и describe.
 Найдите таким образом максимальный возраст мужчин расы Amer-Indian-Eskimo."""
print(data.groupby(['race', 'sex'])['age'].describe().loc['Amer-Indian-Eskimo'].loc['Male']['max'])


"""8. Среди кого больше доля зарабатывающих много (>50K): среди женатых или холостых мужчин (признак marital-status)?
 Женатыми считаем тех, у кого marital-status начинается с Married (Married-civ-spouse, Married-spouse-absent или 
 Married-AF-spouse), остальных считаем холостыми."""
men1 = men[men['salary'] == '>50K']
tabl = pd.crosstab(men1['salary'], men1['marital-status'], normalize=True)
print((tabl.loc['>50K', 'Married-AF-spouse'] + tabl.loc['>50K', 'Married-civ-spouse']
       + tabl.loc['>50K', 'Married-spouse-absent']) > 0.5)


"""9. Какое максимальное число часов человек работает в неделю (признак hours-per-week)? 
Сколько людей работают такое количество часов и каков среди них процент зарабатывающих много?"""
print(data['hours-per-week'].describe()['max'])
h = data[data['hours-per-week'] == 99]
print(len(h))
print(h['salary'].value_counts(normalize=True).loc['>50K'])


"""10. Посчитайте среднее время работы (hours-per-week) зарабатывающих мало и много для каждой страны."""
table = data.groupby(['native-country', 'salary'])['hours-per-week'].mean()

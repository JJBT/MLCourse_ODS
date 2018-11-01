import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('dataset/howpop_train.csv')

df.drop(filter(lambda c: c.endswith('_lognorm'), df.columns),
        axis=1,       # axis = 1: столбцы
        inplace=True)  # избавляет от необходимости сохранять датасет

# настройка внешнего вида графиков в seaborn
sns.set_style("dark")
sns.set_palette("RdBu")
sns.set_context("notebook", font_scale=1.5,
                rc={"figure.figsize": (15, 5), "axes.titlesize": 18})

df['published'] = pd.to_datetime(df.published, yearfirst=True)


df['year'] = [d.year for d in df['published']]
df['month'] = [d.month for d in df['published']]
df['day'] = [d.day for d in df['published']]

df['dayofweek'] = [d.isoweekday() for d in df['published']]
df['hour'] = [d.hour for d in df['published']]


"""1. В каком месяце (и какого года) было больше всего публикаций?"""
# max_public = df.pivot_table(index='month',
#                             columns='year',
#                             values='url', aggfunc='count')
# max_public.plot(kind='bar')
# plt.show()

"""2"""
# sns.countplot(x='day', data=df[(df['month'] == 3) & (df['year'] == 2015)], hue='domain')
# plt.show()

# sns.countplot(x='dayofweek', data=df[(df['month'] == 3) & (df['year'] == 2015)], hue='domain')
# plt.show()

"""3. Когда лучше всего публиковать статью?"""
# public = df.groupby(['domain', 'hour'])['comments'].sum()
# public.plot(kind='bar')
# plt.show()
#
# public_1 = df.groupby(['domain', 'hour'])['views'].sum()
# public_1.plot(kind='bar')
# plt.show()

"""4. Кого из топ-20 авторов по числу статей чаще всего минусуют"""
o = df['author'].value_counts()
o = o.head(20)
df = df[df['author'].isin(o.index)]
tb = df.pivot_table(['votes_minus'], ['author'], aggfunc='mean')
tb = tb.sort_values(by='votes_minus', ascending=False)

"""5. Правда ли, что по субботам авторы пишут в основном днём, а по понедельникам — в основном вечером?"""
pn = df.pivot_table(index=df['dayofweek'],
                    columns=df['hour'],
                    values='url', aggfunc='count')
pn.plot(kind='bar')
plt.show()

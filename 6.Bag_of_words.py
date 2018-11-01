"""Самый простой подход называется Bag of Words: создаем вектор длиной в словарь, для каждого слова считаем количество
 вхождений в текст и подставляем это число на соответствующую позицию в векторе"""

from functools import reduce
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


texts = [['i', 'have', 'a', 'cat'],
         ['he', 'have', 'a', 'dog'],
         ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(reduce(lambda x, y: x + y, texts))))


def vectorize(text):
    vector = np.zeros(len(dictionary))
    for i, word in dictionary:
        num = 0
        for w in text:
            if w == word:
                num += 1
        if num:
            vector[i] = num
    return vector

#
# for t in texts:
#     print(vectorize(t))


"""Используя алгоритмы вроде Вag of Words, мы теряем порядок слов в тексте. 
 Чтобы избежать этой проблемы, можно сделать шаг назад и изменить подход к токенизации:
  например, использовать N-граммы (комбинации из N последовательных терминов).
"""

vect = CountVectorizer(ngram_range=(1, 1))
print(vect.fit_transform(['no i have cows', 'i have no cows']).toarray())
# [[1 1 1]
#  [1 1 1]]
print(vect.vocabulary_)
# {'no': 2, 'have': 1, 'cows': 0}

vect.set_params(ngram_range=(1, 2))
print(vect.fit_transform(['no i have cows', 'i have no cows']).toarray())
# [[1 1 1 0 1 0 1]
#  [1 1 0 1 1 1 0]]
print(vect.vocabulary_)
# {'no': 4, 'have': 1, 'cows': 0, 'no have': 6, 'have cows': 2, 'have no': 3, 'no cows': 5}


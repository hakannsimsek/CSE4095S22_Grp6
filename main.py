import json
import re
import string
from timeit import timeit
import scipy.stats

# Chai Square Test
def calculate_chai_square(word_frequency_matrix, number_of_tokens_in_corpus):
    return number_of_tokens_in_corpus * pow((word_frequency_matrix[0][0] * word_frequency_matrix[1][1]) - (word_frequency_matrix[0][1] * word_frequency_matrix[1][0]), 2) / ((word_frequency_matrix[0][0] + word_frequency_matrix[1][0]) * (word_frequency_matrix[0][1] + word_frequency_matrix[1][1]) * (word_frequency_matrix[0][0] + word_frequency_matrix[0][1]) * (word_frequency_matrix[1][0] + word_frequency_matrix[1][1]))


print('Chai Result', calculate_chai_square([[8, 4667], [15820, 14287181]], 14307668))
critical_chai_square_value = scipy.stats.chi2.ppf(1-.05, df=1)

f = open ('./1.json', "r")

def clear_punctuations(payload):
    return re.sub(r'[^\w\s]','',payload)
 
data = json.loads(f.read())
payload = clear_punctuations(data['ictihat'].strip()).split(' ') # We should remove spaces in the text


print('Payload', payload)

wordMap = {}
gramCount = 5
indicator = gramCount // 2


for i in range(indicator, len(payload) - indicator):
    word = payload[i]
    if word not in wordMap:
        wordMap[word] = {
            -2: {},
            -1: {},
            0: {},
            1: {},
            2: {},
        }
    frequencyMap = wordMap[word]
    for j in range(gramCount):
        gram = payload[i - indicator + j]
        if gram in frequencyMap[-indicator + j]:
            frequencyMap[-indicator + j][gram] += 1
        else:
            frequencyMap[-indicator + j][gram] = 1

# ve = wordMap['ve']

# for i in range(-2, 2):
#     for gram in ve[i]:
#         print(i, gram, ve[i][gram])
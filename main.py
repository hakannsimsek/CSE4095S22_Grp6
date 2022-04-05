import json
import re
from numpy import number
import scipy.stats
from os import listdir
from os.path import isfile, join
# from snowballstemmer import TurkishStemmer

# turk_stemmer = TurkishStemmer()

# from nltk.stem.snowball import SnowballStemmer
  
# the stemmer requires a language parameter
# turk_stemmer = SnowballStemmer(language='turkish')


# Chai Square Test
critical_chai_square_value = scipy.stats.chi2.ppf(1-.05, df=1)
print(critical_chai_square_value)

def calculate_chai_square(word_frequency_matrix, number_of_tokens_in_corpus):
    return number_of_tokens_in_corpus * pow((word_frequency_matrix[0][0] * word_frequency_matrix[1][1]) - (word_frequency_matrix[0][1] * word_frequency_matrix[1][0]), 2) / ((word_frequency_matrix[0][0] + word_frequency_matrix[1][0]) * (word_frequency_matrix[0][1] + word_frequency_matrix[1][1]) * (word_frequency_matrix[0][0] + word_frequency_matrix[0][1]) * (word_frequency_matrix[1][0] + word_frequency_matrix[1][1]))

def construct_frequency_matrix_from_word_map(w1, w2, word_map, number_of_tokens_in_corpus):
    # [[w1w2, w1!w2], [w1w2!, w1!w2!]]
    compound_word_count = 0
    if w2 in word_map[w1][1]:
        compound_word_count = word_map[w1][1][w2]
    w1_count = word_map[w1]['count']
    w2_count = word_map[w2]['count']
    frequency_matrix = [[compound_word_count, w2_count - compound_word_count],[w1_count - compound_word_count,number_of_tokens_in_corpus + 2 * compound_word_count - w1_count - w2_count]]
    return frequency_matrix

def get_top_ten_for_chai_square(word_map, number_of_tokens_in_corpus):
    chai_square_list = []
    for w1 in word_map:
        for w2 in word_map:
            if w1 != w2:
                frequency_matrix = construct_frequency_matrix_from_word_map(w1, w2, word_map, number_of_tokens_in_corpus)
                chai_square_value = calculate_chai_square(frequency_matrix, number_of_tokens_in_corpus)
                if chai_square_value != number_of_tokens_in_corpus and chai_square_value > critical_chai_square_value:
                    chai_square_list.append([w1, w2, chai_square_value])
    chai_square_list.sort(key=lambda x: x[2], reverse=True)
    return chai_square_list[:10]

def clear_punctuations(payload):
    return re.sub(r'[^\w\s]','',payload)

def read_all_data_and_get_payloads(path='data'):
    jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
    payloads = []
    for jsonFileName in jsonFileNames:
        payloads.append(read_json_file_and_get_payload(join(path, jsonFileName)))
    return payloads

def read_some_data_and_get_payloads(path='data', number_of_files=10):
    jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
    payloads = []
    for jsonFileName in jsonFileNames[:number_of_files]:
        payloads.append(read_json_file_and_get_payload(join(path, jsonFileName)))
    return payloads

def read_json_file_and_get_payload(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return clear_punctuations(data['ictihat'].strip()).split(' ')
 
# payload = list(map(turk_stemmer.stemWord, clear_punctuations(data['ictihat'].strip()).split(' '))) # We should remove spaces in the text
# payload = read_json_file_and_get_payload('./1.json')
# number_of_tokens_in_corpus = len(payload)
payloads = read_some_data_and_get_payloads()

word_map = {}

def getWordMap(payload):
    gram_count = 5
    indicator = gram_count // 2

    for i in range(indicator, len(payload) - indicator):
        word = payload[i]
        if word not in word_map:
            word_map[word] = {
                -2: {},
                -1: {},
                0: {},
                1: {},
                2: {},
                'count': 0
            }
        frequency_map = word_map[word]
        frequency_map['count'] += 1
        for j in range(gram_count):
            gram = payload[i - indicator + j]
            if gram in frequency_map[-indicator + j]:
                frequency_map[-indicator + j][gram] += 1
            else:
                frequency_map[-indicator + j][gram] = 1
    return len(payload)

number_of_tokens_in_corpus_sum = 0

for payload in payloads:
    number_of_tokens_in_corpus_sum += getWordMap(payload)
print(number_of_tokens_in_corpus_sum)

# print(construct_frequency_matrix_from_word_map('yaralama', 'suçundan', word_map, number_of_tokens_in_corpus))
print(get_top_ten_for_chai_square(word_map, number_of_tokens_in_corpus_sum))

# for i in range(-2, 2):
#     for gram in ve[i]:
#         print(i, gram, ve[i][gram])
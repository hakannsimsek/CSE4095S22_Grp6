import json
import re
import math
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
    return re.sub(r'[^\w\s]' ,'' ,payload)

def read_all_data_and_get_payloads(path='data'):
    jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
    payloads = []
    for jsonFileName in jsonFileNames:
        payloads.append(read_json_file_and_get_payload(join(path, jsonFileName)))
    return payloads

def read_some_data_and_get_payloads(path='data', number_of_files=100):
    jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
    payloads = []
    for jsonFileName in jsonFileNames[:number_of_files]:
        payloads.append(read_json_file_and_get_payload(join(path, jsonFileName)))
    return payloads

def read_data_by_day_and_get_payloads(path='data', day='01'):
    jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
    payloads = []
    for jsonFileName in jsonFileNames:
        data = read_json_file(join(path, jsonFileName))
        if data['Mahkeme Günü'] == day:
            payloads.append(get_payload(data))
    return payloads 

def read_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def get_payload(data):
    payload = clear_punctuations(data['ictihat'].lower().strip()).split(' ')
    return [ele for ele in payload if ele.strip()]

def read_json_file_and_get_payload(file_name):
    return get_payload(read_json_file(file_name))
 
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
        if word.isalpha() and word != "":
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
# print(number_of_tokens_in_corpus_sum)

# print(construct_frequency_matrix_from_word_map('yaralama', 'suçundan', word_map, number_of_tokens_in_corpus))
# print(get_top_ten_for_chai_square(word_map, number_of_tokens_in_corpus_sum))

# for i in range(-2, 2):
#     for gram in ve[i]:
#         print(i, gram, ve[i][gram])

print(read_data_by_day_and_get_payloads(day='04')[0])
list_of_dict_values = list(word_map.items())
statisticOfWordPairs = {}

def my_function(key, pair):
    arr = [0, 0, 0, 0, 0]
    for i in range(-2, 3):
        if i == 0:
            continue
        adjacents = key[1][i]
        if pair in adjacents.keys():
            payir = adjacents[pair]
            arr[i+2] = adjacents[pair]

    #print("key : " ,key )
    #print("pair" , pair)
    mean,count = calculateMean(arr)
    #print("mean : ", mean)
    #if mean % 1 != 0:
        #print("")
    variance = calculateVariance(arr,mean)
    #print("var : ", variance)
    stdDev = math.sqrt(variance)
    #print("std dev : ", stdDev)
    #print("\n")
    return mean,stdDev,arr,count

def calculateVariance(arr,mean):
    sum = 0
    count = 0
    for i in range(0,5):
        count += arr[i]
        sum += arr[i]*( ((i-2) - mean)*((i-2) - mean) )

    if count < 2 :
        return 0
    sSqaure = (sum/(count-1))
    #print("")
    #if sSqaure==-1:
        #print("")

    return sSqaure


def calculateMean(arr):
    sum = 0
    count = 0
    for i in range(0,5):
        count += arr[i]
        sum += arr[i] * (i-2)

    if count <= 1 :
        return 0,count
    mean = sum/count
    return mean,count


countSum = 0
numberOFPairs = 0
for key in list_of_dict_values:
        for i in range(-2, 3):
            #print("here : " , i, key[0], key[1][i])
            #if key[0] == "hem":
            adjacents = key[1][i]
            if i == 0:
                continue
            #if key[0] == "davacı":
            for pair in adjacents:
                ##print("key zero ", key[0])
                mean,stdDev,arr,count = my_function(key, pair)
                if count > 8 and mean > 0.8 and mean < 1.2 and stdDev < 0.3:
                    print("\"",key[0],"\"\"", pair,"\"", " mean : ", mean, " std dev : ", stdDev , arr ,count)
                    countSum += count
                    numberOFPairs += 1
                    #print("countSum : ", countSum, " #pairs : ", numberOFPairs)


                #print("key zero ", key[0])
                if key[0] not in statisticOfWordPairs.keys():
                    statisticOfWordPairs[key[0]] = []
                statisticOfWordPairs[key[0]].append({
                    0: pair,
                    1: mean,
                    2: stdDev,
                    3: arr
                })

                #print(i, pair, adjacents[pair])

print("avg count : ", countSum/numberOFPairs)
print("")

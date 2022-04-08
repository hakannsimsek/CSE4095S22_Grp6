import json
import os
import re
import math
from numpy import number
import scipy.stats
from os import listdir
from os.path import isfile, join
from gram_creator import get_word_map_for_payloads
from extractor import Extractor, stem_word
import math

from nltk import bigrams
from tagger import detectStopWord, tag_word


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

class Pair:
  def __init__(self, focus,adjacent, mean, stdDev, count):
    self.focus = focus
    self.adjacent = adjacent
    self.mean = mean
    self.stdDev = stdDev
    self.count = count

def traverse_payloads_and_print_top_thousand_for_mean(topthousantresult,day,number_of_tokens_in_corpus):
    if number_of_tokens_in_corpus > 0:
        # Create directory if it doesn't exist
        raw_filename = "meanresult/{}.txt".format(day)
        try:
            os.remove(raw_filename)
        except:
            pass
        raw_f = open(raw_filename, 'w')
        print("Top ten for mean and variance:")
        for i in range(len(topthousantresult)):
            w1, w2 = topthousantresult[i][0], topthousantresult[i][1]
            raw_f.write("{} {},{} {}\n".format(w1, w2, stem_word(w1), stem_word(w2)))
        print("Day {} is written in {}".format(day, raw_filename))
        raw_f.close()
        return
    print('No data for this day')


# take the second element for sort
def take_count(pair):
    return pair.count

def mean_and_variance(day, list_of_dict_values, statisticOfWordPairs, number_of_tokens_in_corpus):
    countSum = 0
    numberOFPairs = 0
    NUMBER_OF_TOP = 20
    sett = set()
    for key in list_of_dict_values:
        for i in range(-2, 3):
            # print("here : " , i, key[0], key[1][i])
            # if key[0] == "hem":
            adjacents = key[1][i]
            if i == 0:
                continue
            # if key[0] == "davacı":
            for pair in adjacents:
                ##print("key zero ", key[0])
                mean, stdDev, arr, count = my_function(key, pair)
                if count > 2 and mean > 0 and mean < 2 and stdDev < 1:
                    #print("\"", key[0], "\"\"", pair, "\"", " mean : ", mean, " std dev : ", stdDev, arr, count)
                    countSum += count
                    numberOFPairs += 1
                    # print("countSum : ", countSum, " #pairs : ", numberOFPairs)

                # print("key zero ", key[0])
                if key[0] not in statisticOfWordPairs.keys():
                    statisticOfWordPairs[key[0]] = []
                statisticOfWordPairs[key[0]].append(
                    {
                        0: pair,
                        1: mean,
                        2: stdDev,
                        3: count,
                        4: arr
                    })

                # print(i, pair, adjacents[pair])

    itemsofStats = statisticOfWordPairs.items()
    for key in itemsofStats:
        for i in range(-2, 3):
            adjacent = key[1][i]
            if i == 0 :
                continue
            if adjacent[1] > 0 and adjacent[1] < 2:
                #key[0], adjacent[0]
                if detectStopWord(key[0], adjacent[0]):
                    sett.add(Pair(key[0], adjacent[0], adjacent[1], adjacent[2], adjacent[3]))

    #print("avg count : ", countSum / numberOFPairs)
    liste = sorted(sett, key=take_count, reverse=True)
    if len(liste) < NUMBER_OF_TOP:
        NUMBER_OF_TOP = len(liste)
    else :
        NUMBER_OF_TOP = 20
    newTopHundertList = list()
    for i in range(0, NUMBER_OF_TOP):
        newTopHundertList.append(liste.__getitem__(i))

    hashing = dict()
    listee = list()
    for i in newTopHundertList:
        if i.focus not in hashing.keys():
            listee.append([
                i.focus,
                i.adjacent
            ])
            print(i.focus, i.adjacent, i.mean, i.stdDev, i.count)
        hashing[i.focus] = 1

    traverse_payloads_and_print_top_thousand_for_mean(listee,day, number_of_tokens_in_corpus)
    print("")


def traverse_days_and_write_results():
    for day in range(1,  31):
        print("Day : ", day)
        payloads = Extractor.read_data_by_day_and_get_payloads(day=str(day).zfill(2))
        word_map, number_of_tokens_in_corpus = get_word_map_for_payloads(payloads)
        list_of_dict_values = list(word_map.items())
        statisticOfWordPairs = {}
        mean_and_variance(day, list_of_dict_values, statisticOfWordPairs, number_of_tokens_in_corpus)

traverse_days_and_write_results()


# print("avg count : ", countSum/numberOFPairs)
# print("")
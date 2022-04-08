import math
import os
from gram_creator import get_word_map_for_payloads
from extractor import Extractor, stem_word
import math
from tagger import detectStopWord



class Mutual:
  def __init__(self, focus,adjacent, pmi):
    self.focus = focus
    self.adjacent = adjacent
    self.pmi = pmi



class ColPMI:


    def __generate_bigrams(self):
        token_list = [str(tok) for tok in self.doc]

        bgs = bigrams(token_list)
        return [str(bg) for bg in bgs]

    def __count_word_frequency(self, word):
        freq = 0
        for tok in self.doc:
            if tok.text == word:
                freq = freq + 1
        return freq

    def __count_bigram_frequency(self, x, y, word_map):
        freq = 0
        qwe = word_map[x][1]
        if y in word_map[x][1]:
            freq += word_map[x][1][y]
        if y in word_map[x][-1]:
            freq += word_map[x][-1][y]
        """for bg in self.bgs:
            if bg == k:
                freq = freq + 1"""
        return freq

    def __probability(self, x, n):
        return x / n

    def __pmi(self, P_x, P_y, P_xy):
        try:
            return math.log2(P_xy / (P_x * P_y))
        except:
            return 0

    def PMI(self, x, y, word_map, number_of_tokens_in_corpus):
        bg = f"('{x}', '{y}')"
        # frequency
        n_x = word_map[x][0]
        if y not in word_map:
            return
        n_y = word_map[y][0]

        n_bg = self.__count_bigram_frequency(x,y, word_map)

        # probability
        uzunluk = number_of_tokens_in_corpus
        xuzunluk = word_map[x][0][x]
        yuzunuluk = word_map[y][0][y]
        p_x = xuzunluk/uzunluk
        p_y = yuzunuluk/uzunluk
        p_bg = n_bg / uzunluk #tüm bigramların sayısı (tüm kelimeler sayısı asdsadsadsa -1)

        # pmi
        pmi = self.__pmi(p_x, p_y, p_bg)

        #return [x, y, n_x, n_y, n_bg, p_x, p_y, p_bg, pmi]
        return x,y,pmi
col_pmi = ColPMI()
results = []

# results.append(col_pmi.PMI('davacı', 'adına'))
# results.append(col_pmi.PMI('davacı', 'tarafından'))
# results.append(col_pmi.PMI('sair', 'temyiz'))

def pmi_sort(mutual):
    return mutual.pmi

def traverse_payloads_and_print_top_thousand_for_pmi(topthousantresult,day, number_of_tokens_in_corpus):
    if number_of_tokens_in_corpus > 0:
        # Create directory if it doesn't exist
        raw_filename = "pmiresult/{}.txt".format(day)
        try:
            os.remove(raw_filename)
        except:
            pass
        raw_f = open(raw_filename, 'w')
        print("Top ten for PMI:")
        for mutualObj in topthousantresult:
            w1, w2 = mutualObj.focus, mutualObj.adjacent
            raw_f.write("{} {},{} {}\n".format(w1, w2, stem_word(w1), stem_word(w2)))
        print("Day {} is written in {}".format(day, raw_filename))
        raw_f.close()
        return
    print('No data for this day')

def write_results(payloads, day):
    word_map, number_of_tokens_in_corpus = get_word_map_for_payloads(payloads)
    list_of_dict_values = list(word_map.items())
    max = 0
    kelime = ""
    pair = ""
    TOP_X_NUMBER = 50
    pmiList = set()

    for key in list_of_dict_values:
            i = 1
            # print("here : " , i, key[0], key[1][i])
            # if key[0] == "hem":
            adjacents = key[1][i]
            if i == 0:
                continue
            # if key[0] == "davacı":
            for pair in adjacents:
                #print("key zero ", key[0],pair)
                if col_pmi.PMI(key[0], pair, word_map, number_of_tokens_in_corpus) is not None:
                    focus,adjacent,pmi = col_pmi.PMI(key[0], pair, word_map, number_of_tokens_in_corpus)
                    #print("focus : ", focus)
                    if detectStopWord(focus, adjacent):
                        mutualObject = Mutual(focus,adjacent,pmi)
                        pmiList.add(mutualObject)

    liste = sorted(pmiList, key=pmi_sort, reverse=True)
    topList = list()
    for i in range(0, TOP_X_NUMBER):
        collocation = liste.__getitem__(i)
        topList.append(collocation)
        print(i,collocation.focus,collocation.adjacent,collocation.pmi)
    traverse_payloads_and_print_top_thousand_for_pmi(topList, day, number_of_tokens_in_corpus)
    

def traverse_payloads_and_write_results():
    for day in range(1, 31):
        payloads = Extractor.read_data_by_day_and_get_payloads(day=str(day).zfill(2))
        if len(payloads) > 0:
            write_results(payloads, day)

traverse_payloads_and_write_results()
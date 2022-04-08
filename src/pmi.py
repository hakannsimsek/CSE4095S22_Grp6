import math


class ColPMI:
    def __init__(self, word_map, number_of_tokens_in_corpus_sum):
        self.word_map = word_map
        self.number_of_tokens_in_corpus_sum = number_of_tokens_in_corpus_sum

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

    def __count_bigram_frequency(self, x, y):
        freq = 0
        qwe = self.word_map[x][1]
        if y in self.word_map[x][1]:
            freq += self.word_map[x][1][y]
        if y in self.word_map[x][-1]:
            freq += self.word_map[x][-1][y]
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

    def PMI(self, x, y):
        bg = f"('{x}', '{y}')"
        # frequency
        n_x = self.word_map[x][0]
        if y not in self.word_map:
            return
        n_y = self.word_map[y][0]

        n_bg = self.__count_bigram_frequency(x,y)

        # probability
        uzunluk = self.number_of_tokens_in_corpus_sum
        xuzunluk = self.word_map[x][0][x]
        yuzunuluk = self.word_map[y][0][y]
        p_x = xuzunluk/uzunluk
        p_y = yuzunuluk/uzunluk
        p_bg = n_bg / uzunluk #tüm bigramların sayısı (tüm kelimeler sayısı asdsadsadsa -1)

        # pmi
        pmi = self.__pmi(p_x, p_y, p_bg)

        #return [x, y, n_x, n_y, n_bg, p_x, p_y, p_bg, pmi]
        return pmi

def print_pmi(word_map, number_of_tokens_in_corpus):
    col_pmi = ColPMI(word_map, number_of_tokens_in_corpus)
    results = []

    list_of_dict_values = list(word_map.items())
    max = 0
    kelime = ""
    pair = ""
    for key in list_of_dict_values:
            i = 1
            # print("here : " , i, key[0], key[1][i])
            # if key[0] == "hem":
            adjacents = key[1][i]
            if i == 0:
                continue
            # if key[0] == "davacı":
            for pair in adjacents:
                print("key zero ", key[0],pair)
                value = col_pmi.PMI(key[0], pair)
                if value is not None and max < value:
                    max = value
                    kelime = key[0]
                    pair = pair

    print("sonuc ", kelime , pair)
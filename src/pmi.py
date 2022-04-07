import spacy

import math

# for bigrams
from nltk import bigrams


# In[18]:


spacy.cli.download("en_core_web_sm")


# In[19]:


model = spacy.load('en_core_web_sm')


# In[24]:


def load_corpus():
    with open('deneme.txt') as text_file:
        text = text_file.read()

        return text

doc = model(load_corpus())


# In[31]:


# count the tokens in the doc
n_tokens = len(doc)
print(f"Number of tokens in the corpus: {n_tokens}\n")
# output


# In[32]:


def generate_bigrams(doc):
    token_list = [str(tok) for tok in doc]

    bgs = bigrams(token_list)
    return [str(bg) for bg in bgs]


bgs = generate_bigrams(doc=doc)


# In[36]:


def count_word_frequency(word, doc):
    freq = 0
    for tok in doc:
        if tok.text == word:
            freq = freq + 1

    return freq


n_sunflower = count_word_frequency(word='sunflower', doc=doc)
n_seed = count_word_frequency(word='seed', doc=doc)


def count_bigram_frequency(k, bigrams):
    freq = 0
    for bg in bigrams:
        if bg == k:
            freq = freq + 1
    return freq


n_sunflower_seed = count_bigram_frequency(
    k="('sair', 'temyiz')", bigrams=bgs)

print(
    f"sair = {n_sunflower}\ntemyiz = {n_seed}\nsair temyiz = {n_sunflower_seed}\n")


# In[39]:


def probability(x, n):
    return x / n


def pmi(P_x, P_y, P_xy):
    return math.log2(P_xy / (P_x * P_y))
p_sunflower = probability(n_sunflower, n_tokens)
p_seed = probability(n_seed, n_tokens)
p_sunflower_seed = probability(n_sunflower_seed, len(bgs))

r = pmi(p_sunflower, p_seed, p_sunflower_seed)

print(f"pmi for sair temyiz = {r}")
# output


# In[46]:


import spacy
import math

from nltk import bigrams
from tabulate import tabulate


class ColPMI:
    def __init__(self):
        self.model = spacy.load('en_core_web_sm')
        self.doc = self.model(self.__load_corpus())
        self.n_tokens = len(self.doc)
        self.bgs = self.__generate_bigrams()

    def __load_corpus(self):
        return read_data_by_day_and_get_payloads().join(" ")

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

    def __count_bigram_frequency(self, k):
        freq = 0
        for bg in self.bgs:
            if bg == k:
                freq = freq + 1
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
        n_x = self.__count_word_frequency(word=x)
        n_y = self.__count_word_frequency(word=y)
        n_bg = self.__count_bigram_frequency(k=bg)

        # probability
        p_x = self.__probability(n_x, self.n_tokens)
        p_y = self.__probability(n_y, self.n_tokens)
        p_bg = self.__probability(n_bg, len(self.bgs))

        # pmi
        pmi = self.__pmi(p_x, p_y, p_bg)

        return [x, y, n_x, n_y, n_bg, p_x, p_y, p_bg, pmi]
col_pmi = ColPMI()
results = []

results.append(col_pmi.PMI('davacı', 'vekili'))
results.append(col_pmi.PMI('davacı', 'adına'))
results.append(col_pmi.PMI('davacı', 'tarafından'))
results.append(col_pmi.PMI('sair', 'temyiz'))


print(tabulate(results, headers=[
      'x', 'y', 'C(x)', 'C(y)', 'C(x, y)', 'P(x)', 'P(y)', 'P(x, y)', 'PMI'], tablefmt='orgtbl'))


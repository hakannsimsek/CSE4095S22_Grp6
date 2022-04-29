from ntpath import join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from reader import Extractor
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

countvectorizer = CountVectorizer(analyzer= 'word')
tfidfvectorizer = TfidfVectorizer(analyzer='word')
# convert th documents into a matrix
count_wm = countvectorizer.fit_transform(corpus)
tfidf_wm = tfidfvectorizer.fit_transform(corpus)
#retrieve the terms found in the corpora
# if we take same parameters on both Classes(CountVectorizer and TfidfVectorizer) , it will give same output of get_feature_names() methods)
#count_tokens = tfidfvectorizer.get_feature_names() # no difference
count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_countvect = pd.DataFrame(data = count_wm.toarray(),index = ['Doc1','Doc2','Doc3', 'Doc4'],columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = ['Doc1','Doc2','Doc3', 'Doc4'],columns = tfidf_tokens)
print("Count Vectorizer\n")
print(df_countvect)
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)

crimeMap = Extractor.read_all_data_and_get_crime_and_corpus()
sort_orders = sorted(crimeMap.items(), key=lambda x: x[1], reverse=True)
print(sort_orders)

# crimeCorpusMap = {}

# for i in range(1,1):
#     jsonFileName = 'data/' + str(i) + '.json'
#     content = Extractor.read_json_file(jsonFileName)
#     crimeName = content['Suç']
#     if crimeName == '':
#         if 'undefined' not in crimeCorpusMap:
#             crimeCorpusMap['undefined'] = 0
#         crimeName = 'undefined'
#     if content['Suç'] not in crimeCorpusMap:
#         crimeCorpusMap[content['Suç']] = 0
#     print(crimeName)
#     # crimeCorpusMap[content['Suç']] = crimeCorpusMap[content['Suç']].append(content['ictihat'])
#     crimeCorpusMap[crimeName] = crimeCorpusMap[crimeName] + 1


# print(crimeCorpusMap)
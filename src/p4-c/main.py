from ntpath import join

from keras.applications.densenet import layers
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split
from reader import Reader
from sklearn import datasets, svm
import codecs
from idlelib import history

import seaborn as sns
from keras.preprocessing.text import Tokenizer

import nltk
import numpy as np
import pandas as pd
from comtypes.safearray import numpy
from keras import Sequential, regularizers
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Flatten, MaxPooling1D, GlobalMaxPooling1D, Dropout, \
    LSTM
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from nltk import RegexpTokenizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import utils
from tensorflow import optimizers, keras
from tqdm import tqdm

from os import listdir
from os.path import isfile, join
import json
import re
from snowballstemmer import TurkishStemmer

turkStem = TurkishStemmer()

import os
from gensim.models.fasttext import FastText

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


def extract_features(df, training_data, testing_data, type="binary"):
    field = 'Mahkemesi'
    print("Extracting features and creating vocabulary...")

    if "binary" in type:

        cv = CountVectorizer(binary=True, max_df=0.95)
        cv.fit_transform(training_data[field].values)

        train_feature_set = cv.transform(training_data[field].values)
        test_feature_set = cv.transform(testing_data[field].values)

        return train_feature_set, test_feature_set, cv

    elif "counts" in type:

        cv = CountVectorizer(binary=False, max_df=0.95)
        cv.fit_transform(training_data[field].values)

        train_feature_set = cv.transform(training_data[field].values)
        test_feature_set = cv.transform(testing_data[field].values)

        return train_feature_set, test_feature_set, cv

    else:

        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_vectorizer.fit_transform(training_data[field].values)

        train_feature_set = tfidf_vectorizer.transform(training_data[field].values)
        test_feature_set = tfidf_vectorizer.transform(testing_data[field].values)

        return train_feature_set, test_feature_set, tfidf_vectorizer


def get_count_vm_and_features(docs):
    vectorizer = CountVectorizer(binary=True, max_features=5000)
    count_vm = vectorizer.fit_transform(docs)
    return count_vm, vectorizer.get_feature_names_out()


def get_tfidf_vm_and_features(docs, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_vectorizer = vectorizer.fit_transform(docs)
    return tfidf_vectorizer, vectorizer.get_feature_names_out()


def print_vectorizer_result(tag, vm, features):
    print('{} Vectorizer\n'.format(tag))
    print(pd.DataFrame(data=vm.toarray(), columns=features, index=range(vm.shape[0])))


def construct_y_by_doc_court_list(doc_court_list):
    y = doc_court_list
    return y


def print_court_map_by(court_map, by='count'):
    for crime in court_map:
        print('{} - {}'.format(crime, court_map[crime][by]))

def split_list(list, portion=0.9):
    return list[:int(len(list) * portion)], list[int(len(list) * portion):]

def apply_fasttext_label_then_get_result_string(class_name, doc):
    return '__label__{} {}'.format(class_name, doc)

def create_file_if_not_exist(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w').close()

# Create directory if not exist
def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def join_with_underscore(string):
    return '_'.join(string.split(' '))

def prepare_data_for_fasttext(docs, doc_court_list):
    [docs_train, docs_test] = split_list(docs)
    [doc_court_list_train, doc_court_list_test] = split_list(doc_court_list)
    doc_court_list_train = list(map(join_with_underscore, doc_court_list_train))
    doc_court_list_test = list(map(join_with_underscore, doc_court_list_test))
    create_dir_if_not_exist('fasttext')
    create_file_if_not_exist('fasttext/train.txt')
    create_file_if_not_exist('fasttext/test.txt')
    with open('fasttext/train.txt', 'w') as f:
        for doc in docs_train:
            f.write(apply_fasttext_label_then_get_result_string(doc_court_list_train[docs_train.index(doc)], doc))
            f.write('\n')
    with open('fasttext/test.txt', 'w') as f:
        for doc in docs_test:
            f.write(apply_fasttext_label_then_get_result_string(doc_court_list_test[docs_test.index(doc)], doc))
            f.write('\n')
    print('Fasttext results are ready')


# This area is common DO NOT change it
print('Obtaining necessary data values...')
court_map, docs, doc_court_list = Reader.read_all_data_and_get_court_map_and_docs_and_doc_court_list()
prepare_data_for_fasttext(docs, doc_court_list)
# Training the fastText classifier
model = fasttext.train_supervised('fasttext/train.txt', wordNgrams = 2)
# Evaluating performance on the entire test file
print('Test Result: ', model.test('fasttext/test.txt'))



# This area is common DO NOT change it
print('Obtaining necessary data values...')
court_map, docs, doc_court_list = Reader.read_all_data_and_get_court_map_and_docs_and_doc_court_list()
print_court_map_by(court_map, by='count')
tfidf_vm, tfidf_features = get_tfidf_vm_and_features(docs)
# print_vectorizer_result('TF-IDF', tfidf_vm, tfidf_features)
X = tfidf_vm.toarray()
y = construct_y_by_doc_court_list(doc_court_list)
print('Splitting data for training and testing...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# End of the common area


# load embeddings
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('wiki.tr.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

sns.set_style("whitegrid")
np.random.seed(0)

sentences = pd.Series(docs)
y = pd.Series(doc_court_list)

sentences_train, sentences_test, train_y, test_y = train_test_split(sentences, y, test_size=0.1, random_state=42)

print("tokenizer")
tokenize = Tokenizer(num_words=1000)
tokenize.fit_on_texts(sentences_train)

X_train = tokenize.texts_to_sequences(sentences_train)
X_test = tokenize.texts_to_sequences(sentences_test)

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print("after pad")

encoder = LabelEncoder()
encoder.fit(train_y)
y_test = encoder.transform(test_y)
y_train = encoder.transform(train_y)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

word_index = tokenize.word_index

embed_dim = 300

# embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print("sample words not found: ", np.random.choice(words_not_found, 10))

def get_top_k_predictions(model, X_test, k):
    probs = model.predict_proba(X_test)

    best_n = np.argsort(probs, axis=1)[:, -k:]

    preds = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    preds = [item[::-1] for item in preds]

    return preds

def collect_preds(Y_test, Y_preds):

    pred_gold_list = [[[Y_test[idx]], pred] for idx, pred in enumerate(Y_preds)]
    return pred_gold_list

def compute_accuracy(eval_items: list):
    correct = 0
    total = 0

    for item in eval_items:
        true_pred = item[0]
        machine_pred = set(item[1])

        for suc in true_pred:
            if suc in machine_pred:
                correct += 1
                break

    accuracy = correct / float(len(eval_items))
    return accuracy

class Extractor:

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Extractor, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def clear_punctuations(payload):
        return re.sub(r'[^\w\s]', '', payload)

    @staticmethod
    def read_all_data_and_get_crime_and_corpus(path='data'):
        jsonFileNames = [str(i) + '.json' for i in range(1, 100)]
        crimeCorpusMap = {}
        crimeList = []
        for jsonFileName in jsonFileNames:
            name = join(path, jsonFileName)
            content = Extractor.read_json_file(name)
            crimeList.append(content)
            crimeName = content['Suç']
            if crimeName == '':
                if 'undefined' not in crimeCorpusMap:
                    crimeCorpusMap['undefined'] = 0
                crimeName = 'undefined'
            if content['Suç'] not in crimeCorpusMap:
                crimeCorpusMap[content['Suç']] = 0
            # crimeCorpusMap[content['Suç']] = crimeCorpusMap[content['Suç']].append(content['ictihat'])
            crimeCorpusMap[crimeName] = crimeCorpusMap[crimeName] + 1
        return crimeCorpusMap, crimeList

    @staticmethod
    def read_all_data_and_get_payloads(path='data'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames:
            payloads.append(Extractor.read_json_file_and_get_payload(join(path, jsonFileName)))
        return payloads

    @staticmethod
    def split_crime_then_fetch_first_one(crime):
        return crime.split(',')[0]

    @staticmethod
    def read_all_data_and_get_crime_and_corpus(path='data'):
        jsonFileNames = [str(i) + '.json' for i in range(1, 1000)]
        crimeCorpusMap = {}
        crimeList = []
        for jsonFileName in jsonFileNames:
            name = join(path, jsonFileName)
            content = Extractor.read_json_file(name)
            crimeList.append(content)
            crimeName = content['Suç']
            if crimeName == '':
                if 'undefined' not in crimeCorpusMap:
                    crimeCorpusMap['undefined'] = 0
                crimeName = 'undefined'
            if content['Suç'] not in crimeCorpusMap:
                crimeCorpusMap[content['Suç']] = 0
            # crimeCorpusMap[content['Suç']] = crimeCorpusMap[content['Suç']].append(content['ictihat'])
            crimeCorpusMap[crimeName] = crimeCorpusMap[crimeName] + 1
        return crimeCorpusMap, crimeList

    @staticmethod
    def read_all_data_and_get_crime_map_and_docs_and_doc_crime_list(path='data'):
        jsonFileNames = [str(i) + '.json' for i in range(1, 10000)]
        crimeCorpusMap = {}
        doc_crime_list = []
        docs = []
        for jsonFileName in jsonFileNames:
            content = Extractor.read_json_file(join(path, jsonFileName))
            crimeName = Extractor.split_crime_then_fetch_first_one(content['Suç'])
            if crimeName == '':
                if 'undefined' not in crimeCorpusMap:
                    crimeCorpusMap['undefined'] = {'corpus': [], 'count': 0}
                crimeName = 'undefined'
            if crimeName not in crimeCorpusMap:
                crimeCorpusMap[crimeName] = {'corpus': [], 'count': 0}
            doc_crime_list.append(crimeName)
            crimeCorpusMap[crimeName]['corpus'].append(content['ictihat'])
            crimeCorpusMap[crimeName]
            docs.append(content['ictihat'])
            crimeCorpusMap[crimeName]['count'] = crimeCorpusMap[crimeName]['count'] + 1
        return crimeCorpusMap, docs, doc_crime_list

    @staticmethod
    def read_some_data_and_get_payloads(path='data', number_of_files=100):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames[:number_of_files]:
            payloads.append(Extractor.read_json_file_and_get_payload(join(path, jsonFileName)))
        return payloads

    @staticmethod
    def read_data_by_day_and_get_payloads(path='data', day='04'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads = []
        for jsonFileName in jsonFileNames:
            data = Extractor.read_json_file(join(path, jsonFileName))
            if data['Mahkeme Günü'] == day:
                payloads.append(Extractor.get_payload(data))
        return payloads

    @staticmethod
    def read_data_by_day_and_get_payloads_day_map(path='data'):
        jsonFileNames = [f for f in listdir(path) if isfile(join(path, f))]
        payloads_day_map = {}
        for jsonFileName in jsonFileNames:
            data = Extractor.read_json_file(join(path, jsonFileName))
            day = data['Mahkeme Günü']
            payloads_day_map[day] = payloads_day_map.get(day, []) + [Extractor.get_payload(data)]
        return payloads_day_map

    @staticmethod
    def read_json_file(filename):
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_payload(data):
        payload = Extractor.clear_punctuations(data['ictihat'].lower().strip()).split(' ')
        return [ele for ele in payload if ele.strip()]

    @staticmethod
    def read_json_file_and_get_payload(file_name):
        return Extractor.get_payload(Extractor.read_json_file(file_name))

    @staticmethod
    def getPlainTextFromPayloads(payloads):
        flatPayload = [item for sublist in payloads for item in sublist]
        return ' '.join(flatPayload)


def stem_word(word):
    return turkStem.stemWord(word)

"""
##logistic regression
crimeMap, crimeList = Extractor.read_all_data_and_get_crime_and_corpus()
df = pd.DataFrame(crimeList)
feature_rep = 'binary'
top = 3
# LogisticR.LogReg(df,feature_rep,top)
training_data, testing_data = train_test_split(df, test_size=0.1, random_state=2000, )
Y_train = training_data['Mahkemesi'].values
Y_test = testing_data['Mahkemesi'].values

X_train, X_test, feature_transformer = extract_features(df, training_data, testing_data, type=feature_rep)

scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear', random_state=2000, C=5, penalty='l2', max_iter=10000)
model = scikit_log_reg.fit(X_train, Y_train)

scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear', random_state=2000, C=5, penalty='l2', max_iter=10000,
                                    class_weight=[embedding_matrix])
model = scikit_log_reg.fit(X_train, Y_train)

preds = get_top_k_predictions(model, X_test, top)

eval_items = collect_preds(Y_test, preds)

accuracy = compute_accuracy(eval_items)
"""

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

# training params
batch_size = 256
num_epochs = 8

# model parameters
num_filters = 64
embed_dim = 300
weight_decay = 1e-4

max_seq_len = 100

# LSTM architecture
modelLSTM = Sequential()
modelLSTM.add(Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=max_seq_len))
modelLSTM.add(LSTM(100))
modelLSTM.add(Dense(10, activation='sigmoid'))
modelLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(modelLSTM.summary())
history = modelLSTM.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=5, batch_size=64)

loss, accuracy = modelLSTM.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = modelLSTM.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

print("training CNN ...")
modelCNN = Sequential()
modelCNN.add(Embedding(nb_words, embed_dim,
                   weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
modelCNN.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
modelCNN.add(MaxPooling1D(2))
modelCNN.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
modelCNN.add(GlobalMaxPooling1D())
modelCNN.add(Dropout(0.5))
modelCNN.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
modelCNN.add(Dense(num_classes, activation='sigmoid'))  # multi-label (k-hot encoding)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
modelCNN.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
modelCNN.summary()

print("model fitting")
history = modelCNN.fit(X_train, y_train,
                   batch_size=32,
                   epochs=15,
                   validation_data=(X_test, y_test))


loss, accuracy = modelCNN.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = modelCNN.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")




print("a")
print("a")
print("a")
print("a")
print("a")
print("a")

"""# load embeddings
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('wiki.tr.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))"""

# sentences = pd.Series(payloads)


print("")
print("")

# SVM Part Start
# Works so slow and that's why: https://stackoverflow.com/questions/40077432/why-is-scikit-learn-svm-svc-extremely-slow
# print('Initiating Support Vector Machine Algorithm...')
# linearSVCLF = svm.LinearSVC(max_iter=1500).fit(X_train, y_train)
# print('Linear SVC Score', linearSVCLF.score(X_test, y_test))
# print('Linear SVC Report', classification_report(y_test, linearSVCLF.predict(X_test)))
# polySVCLF = svm.SVC(kernel='poly', C=1, max_iter=1000).fit(X_train, y_train)
# print('Polynomial SVC Score', polySVCLF.score(X_test, y_test))
# SVM Part End

# Multinomial Naive Bayes Part Start
# print('Initiating Multinomial Naive Bayes Algorithm...')
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB(alpha=1)
# clf.fit(X_train, y_train)
# print('Multinomial Naive Bayes Score', clf.score(X_test, y_test))
# Multinomial Naive Bayes Part End

# y_predicted = clf.predict(X_test)
# cr = classification_report(y_test, y_predicted)
# print(cr)

## Logistic Regression
# crimeMap,crimeList = Reader.read_all_data_and_get_crime_and_corpus()
# df = pd.DataFrame(crimeList)
# feature_rep='binary'
# top=3
# LogisticR.LogReg(df,feature_rep,top)

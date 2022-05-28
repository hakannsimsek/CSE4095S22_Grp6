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
from tensorflow import optimizers
from tqdm import tqdm

from extractor import Extractor
from gram_creator import get_word_map_for_payloads
import os
from extractor import stem_word
from gensim.models.fasttext import FastText
from tagger import create_top_fifty

"""def traverse_payloads_and_print_top_thousand_for_chai_square(day, payloads):
    word_map, number_of_tokens_in_corpus = get_word_map_for_payloads(payloads)
    if number_of_tokens_in_corpus > 0:
        top_thousand_for_chai_square = get_top_thousand_for_chai_square(word_map, number_of_tokens_in_corpus) # [w1, w2, value]
        # Create directory if it doesn't exist
        raw_filename = "chai_square_results/{}.txt".format(day)
        try:
            os.remove(raw_filename)
        except:
            pass
        raw_f = open(raw_filename, 'a')
        print("Top ten for chai square:")
        for i in range(len(top_thousand_for_chai_square)):
            w1, w2 = top_thousand_for_chai_square[i][0], top_thousand_for_chai_square[i][1]
            raw_f.write("{} {},{} {}\n".format(w1, w2, stem_word(w1), stem_word(w2)))
        print("Day {} is written in {}".format(day, raw_filename))
        raw_f.close()
        return
    print('No data for this day')
"""

def main():
    if not os.path.exists(os.path.dirname("/chai_square_results")):
        os.makedirs(os.path.dirname("/chai_square_results"))

    payloads,crimes = Extractor.read_some_data_and_get_payloads_and_crimes('data', 10000)
    print(payloads[0])
    print(crimes[0])

    #------S

    sns.set_style("whitegrid")
    np.random.seed(0)


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


    #-----E

    sentences = pd.Series(payloads)
    y = pd.Series(crimes)

    sentences_train, sentences_test, train_y, test_y = train_test_split(sentences, y, test_size=0.1, random_state=42)

    #------S


    #------E

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

    # training params
    batch_size = 256
    num_epochs = 8

    # model parameters
    num_filters = 64
    embed_dim = 300
    weight_decay = 1e-4

    max_seq_len = 100

    # LSTM architecture
    model = Sequential()
    model.add(Embedding(nb_words, embed_dim, weights=[embedding_matrix],input_length=max_seq_len))
    model.add(LSTM(100))
    model.add(Dense(88, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=5, batch_size=64)



    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
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

    # CNN architecture
    """print("training CNN ...")
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
    plot_graphs(history, "loss")"""


    #----------



main()
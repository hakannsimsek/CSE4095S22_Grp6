from ntpath import join
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split
from reader import Reader
from sklearn import datasets, svm
<<<<<<< HEAD
import fasttext
=======
>>>>>>> 2e8e9e6b1ef2f537b9fa566ee3f5aec02dfb1766

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

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

def prepare_data_for_fasttext(docs, doc_court_list):
    [docs_train, docs_test] = split_list(docs)
    [doc_court_list_train, doc_court_list_test] = split_list(doc_court_list)
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
<<<<<<< HEAD
# Training the fastText classifier
model = fasttext.train_supervised('fasttext/train.txt', wordNgrams = 2)
# Evaluating performance on the entire test file
print('Test Result: ', model.test('fasttext/test.txt'))



# Predicting on a single input
## model.predict(ds.iloc[2, 0])
=======
>>>>>>> 2e8e9e6b1ef2f537b9fa566ee3f5aec02dfb1766
# print_court_map_by(court_map, by='count')
# tfidf_vm, tfidf_features = get_tfidf_vm_and_features(docs)
# print_vectorizer_result('TF-IDF', tfidf_vm, tfidf_features)
# X = tfidf_vm.toarray()
# y = construct_y_by_doc_court_list(doc_court_list)
# print('Splitting data for training and testing...')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# End of the common area

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
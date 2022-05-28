from ntpath import join
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split
from reader import Reader
from sklearn import datasets, svm
import LogisticR

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

def get_top_n_crime_names(crime_map, n=10, reverse=True):
    sorted_crime_list = sorted(crime_map.items(), key=lambda x: x[1]['count'], reverse=reverse)
    first_n_crime_list = sorted_crime_list[:n]
    first_n_crime_names = []
    for crime in first_n_crime_list:
        first_n_crime_names.append(crime[0])
    return first_n_crime_names

def construct_y_by_doc_crime_list(doc_crime_list):
    y = doc_crime_list
    return y

def print_crime_map_by(crime_map, by='count'):
    for crime in crime_map:
        print('{} - {}'.format(crime, crime_map[crime][by]))

def get_optimized_crime_map(crime_map, top_n_crime_names):
    others_key = 'Others'
    optimized_crime_map = { 'Others': { 'corpus': [], 'count': 0 } }
    for index, crimeName in enumerate(crime_map):
        if crimeName in top_n_crime_names:
            optimized_crime_map[crimeName] = crime_map[crimeName]
        else:
            optimized_crime_map[others_key]['corpus'] += crime_map[crimeName]['corpus']
            optimized_crime_map[others_key]['count'] += crime_map[crimeName]['count']
    return optimized_crime_map

def get_optimized_doc_crime_list(doc_crime_list, top_n_crime_names):
    outlier_key = 'Others'
    optimized_doc_crime_list = []
    for crimeName in doc_crime_list:
        if crimeName not in top_n_crime_names:
            optimized_doc_crime_list.append(outlier_key)
        else:
            optimized_doc_crime_list.append(crimeName)
    return optimized_doc_crime_list


# This area is common DO NOT change it
print('Obtaining necessary data values...')
crime_map, docs, doc_crime_list = Reader.read_all_data_and_get_crime_map_and_docs_and_doc_crime_list()
top_n_crime_names = get_top_n_crime_names(crime_map, n=100)
optimized_crime_map = get_optimized_crime_map(crime_map, top_n_crime_names)
optimized_doc_crime_list = get_optimized_doc_crime_list(doc_crime_list, top_n_crime_names=top_n_crime_names)
print_crime_map_by(optimized_crime_map, by='count')
tfidf_vm, tfidf_features = get_tfidf_vm_and_features(docs)
print_vectorizer_result('TF-IDF', tfidf_vm, tfidf_features)
X = tfidf_vm.toarray()
y = construct_y_by_doc_crime_list(optimized_doc_crime_list)
print('Splitting data for training and testing...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# End of the common area

# SVM Part Start
# Works so slow and that's why: https://stackoverflow.com/questions/40077432/why-is-scikit-learn-svm-svc-extremely-slow
print('Initiating Support Vector Machine Algorithm...')
linearSVCLF = svm.LinearSVC(max_iter=1500).fit(X_train, y_train)
print('Linear SVC Score', linearSVCLF.score(X_test, y_test))
# print('Linear SVC Report', classification_report(y_test, linearSVCLF.predict(X_test)))
# polySVCLF = svm.SVC(kernel='poly', C=1, max_iter=1000).fit(X_train, y_train)
# print('Polynomial SVC Score', polySVCLF.score(X_test, y_test))
# SVM Part End

# Multinomial Naive Bayes Part Start
print('Initiating Multinomial Naive Bayes Algorithm...')
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1)
clf.fit(X_train, y_train)
print('Multinomial Naive Bayes Score', clf.score(X_test, y_test))
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
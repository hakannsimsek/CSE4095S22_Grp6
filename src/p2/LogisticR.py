import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import numpy as np
import logging



def LogReg(df,feature_rep,top):

    model,transformer,accuracy=train_model(df, feature_rep, top)
    print("\nAccuracy={0}".format(accuracy))


def extract_features(df, training_data, testing_data, type="binary"):
    field = 'Suç'
    logging.info("Extracting features and creating vocabulary...")

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


def get_top_k_predictions(model, X_test, k):
    probs = model.predict_proba(X_test)

    best_n = np.argsort(probs, axis=1)[:, -k:]

    preds = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    preds = [item[::-1] for item in preds]

    return preds


def train_model(df, feature_rep="binary", top=3):

    training_data, testing_data = train_test_split(df, test_size=0.1,random_state=2000, )

    Y_train = training_data['Suç'].values
    Y_test = testing_data['Suç'].values

    X_train, X_test, feature_transformer = extract_features(df, training_data, testing_data, type=feature_rep)

    scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear', random_state=2000, C=5, penalty='l2', max_iter=10000)
    model = scikit_log_reg.fit(X_train, Y_train)

    preds = get_top_k_predictions(model, X_test, top)

    eval_items = collect_preds(Y_test, preds)

    accuracy = compute_accuracy(eval_items)

    return model, feature_transformer, accuracy


def _reciprocal_rank(true_labels: list, machine_preds: list):

    tp_pos_list = [(idx + 1) for idx, r in enumerate(machine_preds) if r in true_labels]

    rr = 0
    if len(tp_pos_list) > 0:
        first_pos_list = tp_pos_list[0]

        rr = 1 / float(first_pos_list)

    return rr

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
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import pandas as pd
import unicodedata
import random
import re
from tld import get_tld
from urllib.parse import urlparse
# ML algoritms
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support

import pickle


def extract_protocol(url: str) -> str:
    """
    :param url: url que contiene o no el protocolo http(s)
    :return: la url sin el protocolo
    """
    url_result = ""
    try:
        if url.startswith("https://"):
            url_result = url[8:]
        elif url.startswith("http://"):
            url_result = url[7:]
        else:
            url_result = url
    except Exception as e:
        print(e)

    return url_result


def get_tokens(input):
    tokens_by_slash = str(input.encode('utf-8')).split('/')  # get tokens after splitting by slash
    all_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')  # get tokens after splitting by dash
        tokens_by_dot = []
        for j in range(0, len(tokens)):
            temp_tokens = str(tokens[j]).split('.')  # get tokens after splitting by dot
            tokens_by_dot = tokens_by_dot + temp_tokens
        all_tokens = all_tokens + tokens + tokens_by_dot
    all_tokens = list(set(all_tokens))  # remove redundant tokens
    if 'com' in all_tokens:
        all_tokens.remove(
            'com')  # removing .com since it occurs a lot of times and it should not be included in our features
    return all_tokens


def Clasifaier():
    urldata = pd.read_csv("./datasets/1_OK.csv")
    urldata['url'] = urldata['url'].apply(lambda i: extract_protocol(i))
    urldata = np.array(urldata)  # converting it into an array
    random.shuffle(urldata)  # shuffling
    y = [d[1] for d in urldata]  # all result
    urls = [d[0] for d in urldata]
    # vectorizer = TfidfVectorizer(tokenizer=get_tokens)
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(urls)
    models = {
        # "dt": DecisionTreeClassifier(max_depth=10),
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=4),
        # "lr": LogisticRegression(max_iter=150),
        # "svc": SVC(kernel='rbf', random_state=0)
    }
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)
    results = {}
    for algo in models:
        clf = models[algo]
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print("%s : %s " % (algo, score))
        results[algo] = score
    winner = max(results, key=results.get)
    print(winner)
    clf = models[winner]
    res_predict = clf.predict(x_test)
    mt = confusion_matrix(y_test, res_predict)
    print(confusion_matrix(y_test, res_predict))
    print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
    print('False negative rate : %f %%' % ((mt[1][0] / float(sum(mt[1])) * 100)))
    return vectorizer, clf


if __name__ == '__main__':
    vectorizer, clf = Clasifaier()
    pickle.dump(Clasifaier(), open(f'./models/model_clasified.pickle', 'wb'))
    # Clasifaier = pickle.load(open('./models/model_advance_rf.pickle', 'rb'))
    # vectorizer, clf = Clasifaier()
    # X_predict = ['wikipedia.com', 'google.com/search=faizanahad', 'pakistanifacebookforever.com/getpassword.php/',
    #              'www.radsport-voggel.de/wp-admin/includes/log.exe', 'ahrenhei.without-transfer.ru/nethost.exe',
    #              'www.itidea.it/centroesteticosothys/img/_notes/gum.exe']
    #
    # X_predict = vectorizer.transform(X_predict)
    # y_Predict = clf.predict(X_predict)
    # print(y_Predict)

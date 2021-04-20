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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import pickle
from datetime import datetime
now = datetime.now()


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


def add_protocol(url: str) -> str:
    """
    :param url: url que contiene o no el protocolo http(s)
    :return: la url con el protocolo http
    """
    url_result = ""
    try:
        if url.startswith("http://") or url.startswith("https://"):
            url_result = url
        else:
            url_result = f"http://{url}"
    except Exception as e:
        print(e)

    return url_result


def fd_length(url):
    """

    :param url:
    :return:
    """
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except Exception as e:
        return 0


def tld_length(tld):
    """

    :param tld:
    :return:
    """
    try:
        return len(tld)
    except:
        return -1


def digit_count(url):
    """

    :param url:
    :return:
    """
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


def letter_count(url):
    """

    :param url:
    :return:
    """
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


def no_of_dir(url):
    """

    :param url:
    :return:
    """
    urldir = urlparse(url).path
    return urldir.count('/')


def having_ip_address(url):
    """

    :param url:
    :return:
    """
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        # IPv4 in hexadecimal
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1


def shortening_service(url):
    """

    :param url:
    :return:
    """
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1


def get_tokens(input):
    # get tokens after splitting by slash
    tokens_by_slash = str(input.encode('utf-8')).split('/')
    all_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')  # get tokens after splitting by dash
        tokens_by_dot = []
        for j in range(0, len(tokens)):
            # get tokens after splitting by dot
            temp_tokens = str(tokens[j]).split('.')
            tokens_by_dot = tokens_by_dot + temp_tokens
        all_tokens = all_tokens + tokens + tokens_by_dot
    all_tokens = list(set(all_tokens))  # remove redundant tokens
    if 'com' in all_tokens:
        all_tokens.remove(
            'com')  # removing .com since it occurs a lot of times and it should not be included in our features
    return all_tokens


def url_eda(urldata, tipo='basic'):
    """
    Extrae las características según el tipo de análisis
    :param urldata: dataframe
    :param tipo: string
    :return:
    """
    # # 1.1 Length Features
    # # Fix NFKC data for urlparse
    urldata['url'] = urldata['url'].apply(
        lambda i: unicodedata.normalize("NFKC", i))
    if tipo == 'basic':
        urldata['url'] = urldata['url'].apply(lambda i: add_protocol(i))
        # Length of URL
        urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))
        # Hostname Length
        urldata['hostname_length'] = urldata['url'].apply(
            lambda i: len(urlparse(i).netloc))
        # # # Path Length
        urldata['path_length'] = urldata['url'].apply(
            lambda i: len(urlparse(i).path))
        urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))
        # Length of Top Level Domain
        urldata['tld'] = urldata['url'].apply(
            lambda i: get_tld(i, fail_silently=True))
        urldata['tld_length'] = urldata['tld'].apply(lambda i: tld_length(i))
        urldata = urldata.drop("tld", 1)
        #
        # 1.2 Count Features
        urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))
        urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))
        urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))
        urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))
        urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))
        urldata['count='] = urldata['url'].apply(lambda i: i.count('='))
        # urldata['count-http'] = urldata['url'].apply(lambda i: i.count('http'))
        # urldata['count-https'] = urldata['url'].apply(lambda i: i.count('https'))
        urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))
        urldata['count-digits'] = urldata['url'].apply(
            lambda i: digit_count(i))
        urldata['count-letters'] = urldata['url'].apply(
            lambda i: letter_count(i))
        urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))
        # 1.3 Binary Features
        urldata['use_of_ip'] = urldata['url'].apply(
            lambda i: having_ip_address(i))
        urldata['short_url'] = urldata['url'].apply(
            lambda i: shortening_service(i))
        # Predictor Variables
        x = urldata[[
            'hostname_length', 'path_length', 'fd_length',
            'tld_length', 'count-', 'count@', 'count?',
            'count%', 'count.', 'count=', 'count-www',
            'count-digits', 'count-letters', 'count_dir', 'use_of_ip'
        ]]
        # Target Variable
        y = urldata['result']
        return x, y


def model_generator(x, y, algoritms):
    """
    El generador de modelo selecion del conjunto de modelos:
    - DecisionTreeClassifier
    - RandomForestClassifier
    - AdaBoostClassifier
    - GradientBoostingClassifier
    - LogisticRegression
    - GaussianNB
    :param x: Matriz de caracteristicas
    :param y: Vector de predicion
    :return: el obj clf segun el modelo ganador
    """
    # Splitting the data into Training and Testing
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.2, random_state=42)

    # standard scaler
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # Ajuste del modelo
    results = {}
    for algo in algoritms:
        clf = algoritms[algo]
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print("%s : %s " % (algo, score))
        results[algo] = score

    # score selected
    winner = max(results, key=results.get)
    model = algoritms[winner]

    return (model, x_test, y_test)


def evaluar_clf(clf, x_test, y_test):

    y_predict = clf.predict(x_test)
    mtc = confusion_matrix(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)
    prs = precision_score(y_test, y_predict)
    rcall = recall_score(y_test, y_predict)
    print("Matriz Confusion", mtc)
    print("Accuracy", acc)
    print("Precision", prs)
    print("Recall", rcall)


if __name__ == '__main__':

    df = pd.read_csv("./datasets/1_OK.csv")
    print(df.groupby(df['result']).size())
    # Contruir modelo con características léxicas basicas
    X_basic, y_basic = url_eda(df)
    algoritms = {
        "dt": DecisionTreeClassifier(max_depth=10),
        "rf": RandomForestClassifier(n_estimators=50, n_jobs=4),
        "svc": SVC(kernel='rbf', random_state=0),
        "ab": AdaBoostClassifier(n_estimators=50),
        "gb": GradientBoostingClassifier(n_estimators=50),
        "lr": LogisticRegression(random_state=0),
        "gnb": GaussianNB(),
    }
    # genear modelo
    model, x_test, y_test = model_generator(X_basic, y_basic, algoritms)
    # evaluar modelo
    evaluar_clf(model, x_test, y_test)

    # salvar el modelo
    date_time = now.strftime("%m-%d-%YT%H")
    pickle.dump(model, open(f'./models/model_{date_time}.pickle', 'wb'))

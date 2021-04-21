from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd
import unicodedata
import numpy as np
import re
from tld import get_tld
from urllib.parse import urlparse
import pickle


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


class UrlDetector:
    def extract_protocol(self, url: str) -> str:
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

    def add_protocol(self, url: str) -> str:
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

    def fd_length(self, url):
        """

        :param url:
        :return:
        """
        urlpath = urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except Exception as e:
            return 0

    def tld_length(self, tld):
        """

        :param tld:
        :return:
        """
        try:
            return len(tld)
        except:
            return -1

    def digit_count(self, url):
        """

        :param url:
        :return:
        """
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits

    def letter_count(self, url):
        """

        :param url:
        :return:
        """
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters

    def no_of_dir(self, url):
        """

        :param url:
        :return:
        """
        urldir = urlparse(url).path
        return urldir.count('/')

    def having_ip_address(self, url):
        """

        :param url:
        :return:
        """
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            # print match.group()
            return -1
        else:
            # print 'No matching pattern found'
            return 1

    def shortening_service(self, url):
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

    def url_eda(self, urldata, tipo='basic'):
        """
        Extrae las caracteristicas segun el tipo de analisis
        :param urldata: dataframe
        :param tipo: string
        :return:
        """
        # # 1.1 Length Features
        # # Fix NFKC data for urlparse
        urldata['url'] = urldata['url'].apply(lambda i: unicodedata.normalize("NFKC", i))
        if tipo == 'basic':
            urldata['url'] = urldata['url'].apply(lambda i: self.add_protocol(i))
            # Length of URL
            urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))
            # Hostname Length
            urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))
            # # # Path Length
            urldata['path_length'] = urldata['url'].apply(lambda i: len(urlparse(i).path))
            urldata['fd_length'] = urldata['url'].apply(lambda i: self.fd_length(i))
            # Length of Top Level Domain
            urldata['tld'] = urldata['url'].apply(lambda i: get_tld(i, fail_silently=True))
            urldata['tld_length'] = urldata['tld'].apply(lambda i: self.tld_length(i))
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
            urldata['count-digits'] = urldata['url'].apply(lambda i: self.digit_count(i))
            urldata['count-letters'] = urldata['url'].apply(lambda i: self.letter_count(i))
            urldata['count_dir'] = urldata['url'].apply(lambda i: self.no_of_dir(i))
            # 1.3 Binary Features
            urldata['use_of_ip'] = urldata['url'].apply(lambda i: self.having_ip_address(i))
            urldata['short_url'] = urldata['url'].apply(lambda i: self.shortening_service(i))
            # Predictor Variables
            x = urldata[[
                'hostname_length', 'path_length', 'fd_length',
                'tld_length', 'count-', 'count@', 'count?',
                'count%', 'count.', 'count=', 'count-www',
                'count-digits', 'count-letters', 'count_dir', 'use_of_ip'
            ]]
            # Target Variable
            # y = urldata['result']
            return x
        else:
            urldata['url'] = urldata['url'].apply(lambda i: self.extract_protocol(i))
            urldata = np.array(urldata)  # converting it into an array
            # print(urldata)
            random.shuffle(urldata)  # shuffling
            # y = [d[1] for d in urldata]  # all result
            urls = [d[0] for d in urldata]
            # print(urls)
            vectorizer = TfidfVectorizer()

            x = vectorizer.fit_transform(urls)
            print(x)
            return x


class Url(BaseModel):
    url: str
    type: str


class UrlOutPredict(BaseModel):
    url: str
    type: str


app = FastAPI()


@app.get("/")
async def root():
    return {"status": "UP"}


TYPE_EDA = "basic"


@app.post("/api/v1/predict", response_model=UrlOutPredict)
async def predict_url(data: Url):
    url_detection = UrlDetector()

    if data.type == "basic":
        df = pd.DataFrame([data.url], columns=['url'])
        # Contruir modelo con caracteristicas lexicas basicas
        x_basic = url_detection.url_eda(df)
        model = pickle.load(open('./models/model_basic_rf.pickle', 'rb'))
        print(model.predict(x_basic))
        if np.int(model.predict(x_basic)[0]) is 0:
            prediction = 'bening'
        else:
            prediction = 'malicius'

        results = UrlOutPredict(
            url=data.url,
            type=prediction
        )
    else:
        # Contruir modelo con caracteristicas lexicas avanzadas
        urls = []
        urls.append(str(url_detection.extract_protocol(data.url)))

        vectorizer = pickle.load(open('./models/vectorizer_advance.pickle', 'rb'))
        model = pickle.load(open('./models/model_advance_rf.pickle', 'rb'))

        x = vectorizer.fit_transform(urls)
        # print(model.predict(x))

        y = vectorizer.transform(urls)
        print("x", model.predict(x))
        print("y", model.predict(y))

        #
        # if np.int(model.predict(x_advance)[0]) is 0:
        #     prediction = 'bening'
        # else:
        #     prediction = 'malicius'
        #
        results = UrlOutPredict(
            url=data.url,
            type='prediction'
        )
    return results

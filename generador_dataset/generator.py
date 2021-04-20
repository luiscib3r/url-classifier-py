import pandas as pd
import psycopg2
import datetime
import os
from os.path import join, getsize
URL_POSTGRES = "postgresql://urls:urls@localhost:5432/urlsdb"
CREATE_TABLE_URL_DATASET = """
--create  extension "uuid-ossp";
CREATE TABLE IF NOT EXISTS urls_datasets (
    url text not null,
    result int not null,
);
"""


def insert_dt_postgres(url, result):
    pgclient = psycopg2.connect(URL_POSTGRES)
    cur = pgclient.cursor()
    cur.execute(CREATE_TABLE_URL_DATASET)
    cur.execute("INSERT INTO urls_datasets (url,result) VALUES (%s,%s,%s)",
                (url, result)
                )
    pgclient.commit()
    cur.close()
    pgclient.close()


def change_category(cat):
    if cat == 'bad':
        return 1
    elif cat == 'good':
        return 0


def lib_pd_parser():
    # df = pd.read_csv('./salvas/1.csv')
    # df = df.iloc[:, [1, 3]]
    # df.to_csv('./salvas/ok_datasets/1_OK.csv', index=False)

    # df = pd.read_csv('./salvas/2.csv')
    # df['result'] = df['label'].apply(lambda i: change_category(i))
    # df = df.iloc[:, [0, 2]]
    # df.to_csv('./salvas/ok_datasets/2_OK.csv', index=False)

    # df = pd.read_csv('./salvas/3.csv')
    # # print(df)
    # df['result'] = df['label'].apply(lambda i: change_category(i))
    # df = df.iloc[:, [0, 2]]
    # df.to_csv('./salvas/ok_datasets/3_OK.csv', index=False)

    # df = pd.read_csv('./salvas/4.csv')
    # df['result'] = df['label'].apply(lambda i: change_category(i))
    # df = df.iloc[:, [0, 2]]
    # print(df)
    # df.to_csv('./salvas/ok_datasets/4_OK.csv', index=False)

    # df = pd.read_csv('./salvas/5.csv')
    # df['result'] = df['label']
    # df = df.iloc[:, [0, 2]]
    # # print(df)
    # df.to_csv('./salvas/ok_datasets/5_OK.csv', index=False)

    # df = pd.read_csv('./salvas/6.csv')
    # df['result'] = df['label']
    # df = df.iloc[:, [0, 2]]
    # df.to_csv('./salvas/ok_datasets/6_OK.csv', index=False)

    # df = pd.read_csv('./salvas/7.txt', sep=' ')
    # df['result'] = df['name'].apply(lambda i: 1)
    # df = df.iloc[:, [2, 3]]
    # print(df)
    # df.to_csv('./salvas/ok_datasets/7_OK.csv', index=False)

    # df = pd.read_csv('./salvas/8.csv')
    # df['result'] = df['label']
    # df = df.iloc[:, [0, 2]]
    # # print(df)
    # df.to_csv('./salvas/ok_datasets/8_OK.csv', index=False)

    # df = pd.read_csv('./salvas/9.csv')
    # df['result'] = df['label']
    # df = df.iloc[:, [0, 2]]
    # df.to_csv('./salvas/ok_datasets/9_OK.csv', index=False)

    # df = pd.read_csv('./salvas/10.csv')
    # df['result'] = df['label']
    # df = df.iloc[:, [0, 2]]
    # df.to_csv('./salvas/ok_datasets/10_OK.csv', index=False)

    # df = pd.read_csv('./salvas/11.csv')
    # df['result'] = df['label']
    # df = df.iloc[:, [0, 2]]
    # df.to_csv('./salvas/ok_datasets/11_OK.csv', index=False)

    # df = pd.read_csv('./salvas/12.csv', sep='\n')
    # # print(df)
    # df['result'] = df['url'].apply(lambda i: 1)
    # df = df.iloc[:, [0, 1]]
    # # print(df)
    # df.to_csv('./salvas/ok_datasets/12_OK.csv', index=False)

    # df = pd.read_csv('./salvas/13.csv', sep=',')
    # # print(df)
    # df['result'] = df['url'].apply(lambda i: 1)
    # df = df.iloc[:, [0, 2]]
    # # print(df)
    # df.to_csv('./salvas/ok_datasets/13_OK.csv', index=False)

    # df = pd.read_csv('./salvas/14.csv', sep=',')
    # # print(df)
    # df['result'] = df['url'].apply(lambda i: 1)
    # df = df.iloc[:, [0, 2]]
    # print(df)
    # df.to_csv('./salvas/ok_datasets/13_OK.csv', index=False)
    pass

DATASETS = '/Users/qwerty/PycharmProjects/tesis/http_predictor/src/generador_dataset/salvas/ok_datasets'

# nota 0 beningn 1 malicious
if __name__ == '__main__':
    full_name_csv = []
    for root, dirs, files in os.walk(DATASETS):
        for name in files:
            full_path = os.path.join(root, name)
            with open(full_path, 'r') as f:
                for line in f.readlines():
                    url, result = line.split(',')
                    insert_dt_postgres(url, result)

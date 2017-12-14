#  -*- coding: utf-8 -*-
import os
import csv
import sys
import json
import codecs
import logging
import unicodecsv
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from model import AveragedPerceptron

reload(sys)
sys.setdefaultencoding('utf-8')

class Reader:
    def __init__(self, vocab_size=30000):
        self.paren_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.data_path = os.path.join(self.paren_path, 'data', 'RCV2_merged')
        self.util_path = os.path.join(self.paren_path, 'data', 'utils')
        self.embed_path = os.path.join(self.paren_path, 'data', 'embedding')
        self.topic_set = set(['CCAT', 'ECAT', 'GCAT', 'MCAT'])
        self.topic_code = {'CCAT': 0, 'ECAT':1, 'GCAT':2, 'MCAT':3}
        self.country_code = {'english':'en', 'chinese':'zh', 'french':'fr', 'german':'de', 'italian':'it', 'spanish':'es'}
        self.vocab_size = vocab_size

    def read_idf(self, lang):
        upath = os.path.join(self.util_path, lang)
        with open(os.path.join(upath, 'idf.json'), "r") as f:
            idf_dict = json.load(f)
        return idf_dict

    def load_embed(self, embed_name1, embed_name2, folder_name):
        # code1, code2 = self.country_code[lang1], self.country_code[lang2]
        embed_path = os.path.join(self.embed_path, folder_name)
        # l1 = folder_name.split('.')[0] + '.' + code1
        # l2 = folder_name.split('.')[0] + '.' + code2

        def load_lang_embed(embed_path, embed_name):
            with open(os.path.join(embed_path, embed_name), "r") as f:
                word_dict = {}
                embed_mat = []
                for i, line in enumerate(f):
                    line = line.strip().split(' ', 1)
                    word_dict[line[0].strip()] = i
                    embed_mat += [map(lambda x: float(x), line[1][1:].strip().split())]

            return word_dict, np.asarray(embed_mat, dtype=np.float32)

        word_dict1, embed_mat1 = load_lang_embed(embed_path, embed_name1)
        word_dict2, embed_mat2 = load_lang_embed(embed_path, embed_name2)
        return word_dict1, embed_mat1, word_dict2, embed_mat2

    def vectorize(self, Xs, idf, word_dict, embed_mat):
        if isinstance(embed_mat, list):
            embed_mat = np.asarray(embed_mat, dtype=np.float32)

        vectors = []
        total_vocabs, embed_dim = embed_mat.shape
        for X in Xs:
            vector = np.zeros((len(X), embed_dim))
            pbar = tqdm(X)
            for i, sent in enumerate(pbar):
                cnt = 0
                sent = word_tokenize(sent.strip())
                for wd in sent:
                    wd = wd.lower()
                    if wd in word_dict:
                        if wd in idf:
                            vector[i, :] += idf[wd] * embed_mat[word_dict[wd], :]
                        # else:
                        #     vector[i, :] += embed_mat[word_dict[wd], :]
                        cnt += 1

                pbar.set_description("%d / %d, counted / total words" % (cnt, len(sent)))
                # if i % 1000 == 0: print "%d / %d , counted words / total words" % (cnt, len(sent.split()))

            vectors += [vector]

        return vectors

    def read_files(self, lang):
        in_path = os.path.join(self.data_path, lang)
        def read_file(csv_file_path):
            with open(csv_file_path, "rb") as csvfile:
                csvreader = unicodecsv.reader(csvfile, encoding='utf8')

                header = csvreader.next()
                X, y = [], []
                for row in csvreader:
                    X += [row[2] + " " + row[3]]
                    y += [int(row[1])]
            return X, y

        train_X, train_y = read_file(os.path.join(in_path, "train.csv"))
        test_X, test_y = read_file(os.path.join(in_path, "test.csv"))
        valid_X, valid_y = read_file(os.path.join(in_path, "valid.csv"))

        return train_X, train_y, test_X, test_y, valid_X, valid_y

def test_reading_embed():
    reader = Reader()
    # reader.load_I_matrix_embed('german', 'english', 'de-en.40')

if __name__ == "__main__":
    # reader = Reader()
    # reader.read_files('chinese')
    test_reading_embed()
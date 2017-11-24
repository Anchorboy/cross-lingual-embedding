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
        self.country_code = {'english':'en', 'chineses':'zh', 'french':'fr', 'german':'de', 'italian':'it', 'spanish':'es'}
        self.vocab_size = vocab_size

    def read_idf(self, lang):
        upath = os.path.join(self.util_path, lang)
        with open(os.path.join(upath, 'idf.json'), "r") as f:
            idf_dict = json.load(f)
        return idf_dict

    def load_I_matrix_embed(self, lang1, lang2, folder_name):
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        embed_path = os.path.join(self.embed_path, 'I-Matrix', folder_name)

        l1 = folder_name.split('.')[0] + '.' + code2
        l2 = folder_name.split('.')[0] + '.' + code1

        def load_lang_embed(embed_path, l):
            with open(os.path.join(embed_path, l), "r") as f:
                word_dict = {}
                embed_mat = []
                for i, line in enumerate(f):
                    line = line.strip().split(' ', 1)
                    word_dict[line[0].strip()] = i
                    embed_mat += [map(lambda x: float(x), line[1][1:].strip().split())]

            return word_dict, np.asarray(embed_mat, dtype=np.float32)

        word_dict1, embed_mat1 = load_lang_embed(embed_path, l1)
        word_dict2, embed_mat2 = load_lang_embed(embed_path, l2)
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
                for wd in sent.split():
                    if wd in word_dict:
                        if wd in idf:
                            vector[i, :] += idf[wd] * embed_mat[word_dict[wd], :]
                        else:
                            vector[i, :] += embed_mat[word_dict[wd], :]
                        cnt += 1

                pbar.set_description("%d / %d, counted / total words" % (cnt, len(sent.split())))
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

    def run_I_matrix(self, lang1, lang2, folder_name):
        # --- read files for language 1
        # logger.info('read language1')
        train_X1, train_y1, test_X1, test_y1, valid_X1, valid_y1 = self.read_files(lang1)
        idf1 = self.read_idf(lang1)

        # --- read files for language 2
        # logger.info('read language2')
        train_X2, train_y2, test_X2, test_y2, valid_X2, valid_y2 = self.read_files(lang2)
        idf2 = self.read_idf(lang2)

        # --- read embedding
        # logger.info('read embedding')
        word_dict1, embed_mat1, word_dict2, embed_mat2 = self.load_I_matrix_embed(lang1, lang2, folder_name)

        # --- vectorize language 1
        # logger.info('vectorize language 1')
        train_X1, test_X1, valid_X1 = self.vectorize([train_X1, test_X1, valid_X1], idf1, word_dict1, embed_mat1)
        # test_X1 = self.vectorize(test_X1, idf1, word_dict1, embed_mat1)
        # valid_X1 = self.vectorize(valid_X1, idf1, word_dict1, embed_mat1)

        # --- vectorize language 2
        # logger.info('vectorize language 2')
        train_X2, test_X2, valid_X2 = self.vectorize([train_X2, test_X2, valid_X2], idf1, word_dict1, embed_mat1)
        # train_X2 = self.vectorize(train_X2, idf2, word_dict2, embed_mat2)
        # test_X2 = self.vectorize(test_X2, idf2, word_dict2, embed_mat2)
        # valid_X2 = self.vectorize(valid_X2, idf2, word_dict2, embed_mat2)

        # --- train model



def test_reading_embed():
    reader = Reader()
    # reader.load_I_matrix_embed('german', 'english', 'de-en.40')
    reader.run_I_matrix('german', 'english', 'de-en.40')

if __name__ == "__main__":
    # reader = Reader()
    # reader.read_files('chinese')
    test_reading_embed()
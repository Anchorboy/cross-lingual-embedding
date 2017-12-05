#  -*- coding: utf-8 -*-
import os
import csv
import sys
parendir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parendir)
import json
import shutil
import jieba
import random
import codecs
import pickle
import collections
import numpy as np
from util.config import Config
from scipy import sparse

class Preprocessing:
    def __init__(self):
        self.config = Config
        self.paren_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
        self.data_path = os.path.join(self.paren_path, 'data', 'ted_2015')
        self.util_path = os.path.join(self.paren_path, 'data', 'utils')
        self.country_code = {'english': 'en', 'chinese': 'zh', 'french': 'fr', 'german': 'de', 'italian': 'it',
                             'spanish': 'es'}
        self.lang1 = None
        self.lang2 = None
        self.folder_name = None

    def read_bilingual_data(self, lang1, lang2, folder_name):
        self.lang1 = lang1
        self.lang2 = lang2
        self.folder_name = folder_name
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        in_path = os.path.join(self.data_path, folder_name)
        f1 = os.path.join(in_path, 'train.tags.' + folder_name + '.' + code1)
        f2 = os.path.join(in_path, 'train.tags.' + folder_name + '.' + code2)

        with open(f1, 'r', encoding='utf8') as f:
            text1 = f.readlines()
        with open(f2, 'r', encoding='utf8') as f:
            text2 = f.readlines()

        # 必須是平行的語料
        assert len(text1) == len(text2)
        self.text1 = text1
        self.text2 = text2

    def cut_chinese(self):
        in_path = os.path.join(self.data_path, 'en-zh')
        f1 = os.path.join(in_path, 'train.tags.en-zh.zh')

        out_file = open(f1+'2', "w")
        with open(f1, 'r') as f:
            text = f.readlines()
            for line in text:
                out_file.write(" ".join(jieba.cut(line.strip(), cut_all=False)) + "\n")

    def build_vocab_table(self):
        code1, code2 = self.country_code[self.lang1], self.country_code[self.lang2]
        in_path = os.path.join(self.data_path, self.folder_name)
        f1 = os.path.join(in_path, code1 + '_vocab.json')
        f2 = os.path.join(in_path, code2 + '_vocab.json')

        counter1 = collections.Counter()
        counter2 = collections.Counter()
        for l in self.text1: counter1.update(l.strip().split())
        for l in self.text2: counter2.update(l.strip().split())

        with open(f1, "w", encoding='utf8') as f:
            json.dump(dict([(j[0], i) for i, j in enumerate(counter1.most_common(self.config.vocab_size))]), f)
        with open(f2, "w", encoding='utf8') as f:
            json.dump(dict([(j[0], i) for i, j in enumerate(counter2.most_common(self.config.vocab_size))]), f)

    def dump_to_npy(self):
        code1, code2 = self.country_code[self.lang1], self.country_code[self.lang2]
        in_path = os.path.join(self.data_path, self.folder_name)
        f1 = os.path.join(in_path, 'npy', code1)
        f2 = os.path.join(in_path, 'npy', code2)
        pf = os.path.join(in_path, 'npy', 'params.plk')
        vocab_f1 = os.path.join(in_path, code1 + '_vocab.json')
        vocab_f2 = os.path.join(in_path, code2 + '_vocab.json')
        with open(vocab_f1, "r", encoding='utf8') as f:
            vocab_table1 = json.load(f)
        with open(vocab_f2, "r", encoding='utf8') as f:
            vocab_table2 = json.load(f)

        l1 = []
        for l in self.text1:
            tmp = []
            for wd in l.strip().split():
                if wd in vocab_table1: tmp += [vocab_table1[wd]]
            l1 += [tmp]

        l2 = []
        for l in self.text2:
            tmp = []
            for wd in l.strip().split():
                if wd in vocab_table2: tmp += [vocab_table2[wd]]
            l2 += [tmp]

        assert len(l1) == len(l2)

        ll = int(len(l1) / self.config.mini_batch) + 1 \
            if len(l1) % self.config.mini_batch != 0 \
            else int(len(l1) /  self.config.mini_batch)
        a1 = []
        a2 = []
        cnt = 0
        for i in range(ll):
            t1 = np.zeros(self.config.vocab_size)
            t2 = np.zeros(self.config.vocab_size)
            for j in l1[cnt: cnt + self.config.mini_batch]: t1[j] = 1
            for j in l2[cnt: cnt + self.config.mini_batch]: t2[j] = 1
            cnt += self.config.mini_batch
            a1 += [t1]
            a2 += [t2]
        
        with open(f1 + '_csr.plk', "wb") as f: pickle.dump(sparse.csr_matrix(np.asarray(a1, dtype=np.int32)), f)
        with open(f2 + '_csr.plk', "wb") as f: pickle.dump(sparse.csr_matrix(np.asarray(a2, dtype=np.int32)), f)
        with open(pf, "wb") as f: pickle.dump(self.config, f)
        
    def load_npy(self):
        code1, code2 = self.country_code[self.lang1], self.country_code[self.lang2]
        in_path = os.path.join(self.data_path, self.folder_name)
        f1 = os.path.join(in_path, 'npy', code1)
        f2 = os.path.join(in_path, 'npy', code2)
        pf = os.path.join(in_path, 'npy', 'params.plk')
        with open(pf, "rb") as f: config = pickle.load(f)
        with open(f1 + '_csr.plk', "rb") as f: 
            a1 = pickle.load(f)
            a1 = a1.todense()
        with open(f2 + '_csr.plk', "rb") as f: 
            a2 = pickle.load(f)
            a2 = a2.todense()

        return config, a1, a2

def load_npy(lang1, lang2, folder_name):
    prep = Preprocessing()
    prep.read_bilingual_data(lang1, lang2, folder_name)
    return prep.load_npy()
        
def test():
    prep = Preprocessing()
    prep.read_bilingual_data('english', 'chinese', 'en-zh')
    prep.build_vocab_table()
    prep.dump_to_npy()
    # prep.load_npy()
    
    # prep.cut_chinese()

if __name__ == "__main__":
    test()

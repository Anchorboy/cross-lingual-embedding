#  -*- coding: utf-8 -*-
import os
import csv
import sys
import json
import shutil
import jieba
import random
import codecs
import collections

reload(sys)
sys.setdefaultencoding('utf-8')

class Preprocessing:
    def __init__(self, vocab_size=50000):
        self.paren_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.data_path = os.path.join(self.paren_path, 'data', 'ted_2015')
        self.util_path = os.path.join(self.paren_path, 'data', 'utils')
        self.country_code = {'english': 'en', 'chinese': 'zh', 'french': 'fr', 'german': 'de', 'italian': 'it',
                             'spanish': 'es'}
        self.vocab_size = vocab_size

    def read_bilingual_data(self, lang1, lang2, folder_name):
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        in_path = os.path.join(self.data_path, folder_name)
        f1 = os.path.join(in_path, 'train.tags.' + folder_name + '.' + code1)
        f2 = os.path.join(in_path, 'train.tags.' + folder_name + '.' + code2)

        with open(f1, 'r') as f:
            text1 = [line.decode('utf8', errors='ignore') for line in f.readlines()]
        with open(f2, 'r') as f:
            text2 = [line.decode('utf8', errors='ignore') for line in f.readlines()]

        # 必須是平行的語料
        assert len(text1) == len(text2)
        self.text1 = text1
        self.text2 = text2

    def cut_chinese(self):
        in_path = os.path.join(self.data_path, 'en-zh')
        f1 = os.path.join(in_path, 'train.tags.en-zh.zh')

        out_file = open(f1+'2', "w")
        with open(f1, 'r') as f:
            text = [line.decode('utf8', errors='ignore') for line in f.readlines()]
            for line in text:
                out_file.write(" ".join(jieba.cut(line.strip(), cut_all=False)) + "\n")

    def build_vocab_table(self, lang1, lang2, folder_name):
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        in_path = os.path.join(self.data_path, folder_name)
        f1 = os.path.join(in_path, code1 + '_vocab.json')
        f2 = os.path.join(in_path, code2 + '_vocab.json')

        counter1 = collections.Counter()
        counter2 = collections.Counter()
        for l in self.text1: counter1.update(l.strip().split())
        for l in self.text2: counter2.update(l.strip().split())

        with open(f1, "w") as f:
            json.dump(dict([(j[0], i) for i, j in enumerate(counter1.most_common(self.vocab_size))]), f)
        with open(f2, "w") as f:
            json.dump(dict([(j[0], i) for i, j in enumerate(counter2.most_common(self.vocab_size))]), f)

def test():
    prep = Preprocessing()
    prep.read_bilingual_data('english', 'chinese', 'en-zh')
    prep.build_vocab_table('english', 'chinese', 'en-zh')
    # prep.cut_chinese()

if __name__ == "__main__":
    test()

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
        self.embed_path = os.path.join(self.paren_path, 'data', 'embedding')
        self.country_code = {'english': 'en', 'chinese': 'zh', 'french': 'fr', 'german': 'de', 'italian': 'it',
                             'spanish': 'es'}
        self.vocab_size = vocab_size

    def read_bilingual_data(self, lang1, lang2, folder_name):
        """
        讀取雙語平行語料
        :param lang1:
        :param lang2:
        :param folder_name:
        :return:
        """
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
        """
        中文分詞
        :return:
        """
        in_path = os.path.join(self.data_path, 'en-zh')
        f1 = os.path.join(in_path, 'train.tags.en-zh.zh')

        out_file = open(f1+'2', "w")
        with open(f1, 'r') as f:
            text = [line.decode('utf8', errors='ignore') for line in f.readlines()]
            for line in text:
                out_file.write(" ".join(jieba.cut(line.strip(), cut_all=False)) + "\n")

    def build_vocab_table(self, lang1, lang2, folder_name):
        """
        自己建vocab_table
        :param lang1:
        :param lang2:
        :param folder_name:
        :return:
        """
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

    def build_vocab(self, lang1, lang2, embed_name, folder_name):
        """
        function用來讀取之前BAE給出的embedding以及對應的vocab table
        :param lang1:
        :param lang2:
        :param embed_name:
        :param folder_name:
        :return:
        """
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        embed_path = os.path.join(self.embed_path, embed_name, folder_name)
        input_path = os.path.join(self.data_path, folder_name)
        f1 = os.path.join(input_path, code1 + '_vocab_v0.json')
        f2 = os.path.join(input_path, code2 + '_vocab_v0.json')

        l1 = folder_name.split('.')[0] + '.' + code1
        l2 = folder_name.split('.')[0] + '.' + code2
        def load_lang_embed(embed_path, l):
            with open(os.path.join(embed_path, l), "r") as f:
                word_dict = {}
                # embed_mat = []
                for i, line in enumerate(f):
                    line = line.strip().split(' ', 1)
                    word_dict[line[0].strip()] = i
                    # embed_mat += [map(lambda x: float(x), line[1][1:].strip().split())]
            return word_dict

        vocab1 = load_lang_embed(embed_path, l1)
        vocab2 = load_lang_embed(embed_path, l2)
        with open(f1, "w") as f:
            json.dump(vocab1, f)
        with open(f2, "w") as f:
            json.dump(vocab2, f)




def test():
    prep = Preprocessing()
    # prep.read_bilingual_data('english', 'chinese', 'en-zh')
    # prep.build_vocab_table('english', 'chinese', 'en-zh')
    # prep.build_vocab('english', 'german', 'BAE', 'en-de')
    prep.build_vocab('english', 'french', 'BAE', 'en-fr')
    # prep.cut_chinese()

if __name__ == "__main__":
    test()

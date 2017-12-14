#  -*- coding: utf-8 -*-
import os
import csv
import sys
import json
import shutil
import jieba
import random
import codecs
from scipy import sparse
from nltk.tokenize import word_tokenize
import xml.etree.cElementTree as ET

class Preprocessing:
    def __init__(self):
        self.paren_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
        self.embed_path = os.path.join(self.paren_path, 'data', 'embedding')
        self.data_path = os.path.join(self.paren_path, 'data', 'europarl')
        self.util_path = os.path.join(self.paren_path, 'data', 'utils')
        self.country_code = {'english': 'en', 'chinese': 'zh', 'french': 'fr', 'german': 'de', 'italian': 'it',
                             'spanish': 'es'}
        self.toks = {'start': 0, 'stop': 1, 'unk': 2}
        self.lang1 = None
        self.lang2 = None
        self.folder_name = None
        self.embed_name = None

    def read_bilingual_data_folders(self, lang1, lang2, folder_name1, folder_name2):
        self.lang1 = lang1
        self.lang2 = lang2
        self.folder_name = folder_name1 + '-' + folder_name2
        self.text1 = []
        self.text2 = []
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        input_path1 = os.path.join(self.data_path, folder_name1)
        input_path2 = os.path.join(self.data_path, folder_name2)
        dir_list = os.listdir(input_path1)
        for i in dir_list:
            f1 = os.path.join(input_path1, i)
            f2 = os.path.join(input_path2, i)
            if not os.path.exists(f2): continue

            # print(i)
            with open(f1, 'r', encoding='utf8') as f:
                text1 = f.readlines()
            with open(f2, 'r', encoding='utf8') as f:
                text2 = f.readlines()
            if len(text1) == len(text2):
                self.text1 += text1
                self.text2 += text2
            else:
                print(i, len(text1), len(text2))

        # 必須是平行的語料
        assert len(self.text1) == len(self.text2)

    def output(self, lang, news):
        path = os.path.join(self.paren_path, 'data', 'RCV2')
        if not os.path.exists(path): os.mkdir(path)
        sub_path = os.path.join(path, lang)
        if not os.path.exists(sub_path): os.mkdir(sub_path)

        file_path = os.path.join(sub_path, news['id'] + '.txt')
        with open(file_path, "w") as f:
            # json.dump(news, f)
            f.write(news['id'] + "\n")
            f.write(",".join(news['topics']) + "\n")
            f.write(news['headline'].decode('utf8', errors='ignore') + "\n")
            f.write(news['content'].decode('utf8', errors='ignore') + "\n")

    def prepro(self, lang1, lang2):
        i = 0
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        align_path = os.path.join(self.data_path, 'alignment', code1 + '-' + code2 + '.xml.gz.tmp')
        tree = ET.ElementTree(file=align_path)
        output_path = os.path.join(self.data_path, code1 + '-' + code2)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_f1 = open(os.path.join(output_path, code1 + '-' + code2 + '.' + code1), "w", encoding='utf8')
        output_f2 = open(os.path.join(output_path, code1 + '-' + code2 + '.' + code2), "w", encoding='utf8')
        # all_l1 = []
        # all_l2 = []
        for elem in tree.iterfind('linkGrp'):
            f1 = elem.attrib['fromDoc'].split('.')[0] + '.txt'
            f2 = elem.attrib['toDoc'].split('.')[0] + '.txt'
            try:
                with open(os.path.join(self.data_path, f1), "r", encoding='utf8') as f:
                    l1 = f.readlines()
                with open(os.path.join(self.data_path, f2), "r", encoding='utf8') as f:
                    l2 = f.readlines()

                for sub_elem in elem.iterfind('link'):
                    tgt = sub_elem.attrib['xtargets'].split(';')
                    print(tgt)
                    output_f1.write(l1[int(tgt[0])])
                    output_f2.write(l2[int(tgt[1])])
            except IOError or IndexError:
                print("no such file")
            else:
                print("other error")

        # input_path1 = os.path.join(self.data_path, lang1)
        # input_path2 = os.path.join(self.data_path, lang2)
        # # language
        # for l in os.listdir(input_path1):
        #     sub_path = os.path.join(input_path1, l)
        #     # sub director
        #     for file in os.listdir(sub_path):
        #         if i % 1000 == 0: print(i)
        #         file_path = os.path.join(sub_path, file)
        #         tree = ET.ElementTree(file=file_path)
        #         news = {'id': '', 'headline':'', 'content': '', 'topics': []}
        #         # news info
        #         for elem in tree.iter(tag='newsitem'):
        #             news['id'] = elem.attrib['itemid']
        #         # news title
        #         for elem in tree.iter(tag='headline'):
        #             news['headline'] = elem.text
        #         # news content
        #         for elem in tree.iterfind('text/p'):
        #             news['content'] += elem.text
        #         # news topic
        #         for elem in tree.iterfind(".//codes[@class='bip:topics:1.0']/code"):
        #             news['topics'] += [elem.attrib['code']]
        # 
        #         self.output(lang, news)
        #         i += 1

if __name__ == "__main__":
    prep = Preprocessing()
    lang = ['chinese', 'english', 'french', 'german', 'italian', 'spanish']
    prep.prepro('german', 'english')
#  -*- coding: utf-8 -*-
import os
from os.path import join as pjoin
import csv
import sys
parendir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parendir)
import json
import shutil
import jieba
import pickle
import argparse
import collections
import numpy as np
from util.config import Config
from scipy import sparse
from nltk.tokenize import word_tokenize

def load_npy(args):
    prep = Preprocessing(args)
    # prep.read_bilingual_data()
    return prep.load_npy()

def dump_word_embedding(embed_mat1, embed_mat2, vocab_table1, vocab_table2, model_path, lang1, lang2):
    W_x = embed_mat1
    W_y = embed_mat2
    country_code = {'english': 'en', 'chinese': 'zh', 'french': 'fr', 'german': 'de', 'italian': 'it',
                    'spanish': 'es'}
    code1 = country_code[lang1]
    code2 = country_code[lang2]

    vocab_list1 = sorted(vocab_table1, key=lambda x: vocab_table1[x])
    vocab_list2 = sorted(vocab_table2, key=lambda x: vocab_table2[x])

    f1 = os.path.join(model_path, code1 + '-' + code2 + '.' + code1)
    f2 = os.path.join(model_path, code1 + '-' + code2 + '.' + code2)

    with open(f1, "w", encoding='utf8') as f:
        for i, wd in enumerate(vocab_list1):
            f.write(wd + " | ")
            f.write(" ".join([str(w) for w in W_x[i]]) + "\n")
    
    with open(f2, "w", encoding='utf8') as f:
        for i, wd in enumerate(vocab_list2):
            f.write(wd + " | ")
            f.write(" ".join([str(w) for w in W_y[i]]) + "\n")

class Preprocessing:
    def __init__(self, args):
        self.args = args
        self.config = Config
        # self.paren_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
        self.paren_path = args.data_path
        self.embed_path = os.path.join(self.paren_path, 'embedding')
        self.data_path = os.path.join(self.paren_path, args.data_name)
        self.util_path = os.path.join(self.paren_path, 'utils')
        self.country_code = {'english': 'en', 'chinese': 'zh', 'french': 'fr', 'german': 'de', 'italian': 'it',
                             'spanish': 'es'}
        self.toks = {'start': 0, 'stop': 1, 'unk': 2}
        self.lang1 = args.lang1
        self.lang2 = args.lang2
        self.folder_name = args.folder_name
        self.embed_name = None

    def read_bilingual_data(self):
        # lang1 = self.lang1
        # lang2 = self.lang2
        # folder_name = self.folder_name
        # code1, code2 = self.country_code[lang1], self.country_code[lang2]
        # input_path = os.path.join(self.data_path, folder_name)
        # f1 = os.path.join(input_path, self.args.file1)
        # f2 = os.path.join(input_path, self.args.file2)
        f1 = self.args.file1
        f2 = self.args.file2

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
        input_path = os.path.join(self.data_path, self.folder_name)
        f1 = os.path.join(input_path, code1 + '_vocab.json')
        f2 = os.path.join(input_path, code2 + '_vocab.json')

        counter1 = collections.Counter()
        counter2 = collections.Counter()
        for l in self.text1: counter1.update([wd.lower() for wd in word_tokenize(l.strip())])
        for l in self.text2: counter2.update([wd.lower() for wd in word_tokenize(l.strip())])

        with open(f1, "w", encoding='utf8') as f:
            json.dump(dict([(j[0], i) for i, j in enumerate(counter1.most_common(self.config.vocab_size))]), f)
        with open(f2, "w", encoding='utf8') as f:
            json.dump(dict([(j[0], i) for i, j in enumerate(counter2.most_common(self.config.vocab_size))]), f)

    def build_vocab(self, embed_name, folder_name):
        """
        function用來讀取之前模型給出的embedding以及對應的vocab table
        :param lang1:
        :param lang2:
        :param embed_name:
        :param folder_name:
        :return:
        """
        code1, code2 = self.country_code[self.lang1], self.country_code[self.lang2]
        embed_path = os.path.join(self.embed_path, embed_name, folder_name)
        input_path = os.path.join(self.data_path, folder_name)
        f1 = os.path.join(input_path, code1 + '_vocab.json')
        f2 = os.path.join(input_path, code2 + '_vocab.json')

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
        return vocab1, vocab2

    def dump_word_embedding(self, lang1, lang2, folder_name, model_path, embed_name):
        W_x = np.load(model_path + '/W_x.npy')
        W_y = np.load(model_path + '/W_y.npy')

        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        in_path = os.path.join(self.data_path, folder_name)
        vocab_f1 = os.path.join(in_path, code1 + '_vocab.json')
        vocab_f2 = os.path.join(in_path, code2 + '_vocab.json')
        with open(vocab_f1, "r", encoding='utf8') as f:
            vocab_table1 = json.load(f)
        shutil.copyfile(vocab_f1, os.path.join(model_path, code1 + '_vocab.json'))
        with open(vocab_f2, "r", encoding='utf8') as f:
            vocab_table2 = json.load(f)
        shutil.copyfile(vocab_f2, os.path.join(model_path, code2 + '_vocab.json'))
        print(code1, len(vocab_table1), code2, len(vocab_table2))
        vocab_list1 = sorted(vocab_table1, key=lambda x:vocab_table1[x])
        vocab_list2 = sorted(vocab_table2, key=lambda x: vocab_table2[x])

        output_path = os.path.join(self.embed_path, embed_name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = os.path.join(output_path, folder_name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        f1 = os.path.join(output_path, folder_name + '.' + code1)
        f2 = os.path.join(output_path, folder_name + '.' + code2)

        with open(f1, "w", encoding='utf8') as f:
            for i, wd in enumerate(vocab_list1):
                f.write(wd + " | ")
                f.write(" ".join([str(w) for w in W_x[i]]) + "\n")
        shutil.copyfile(f1, os.path.join(model_path, folder_name + '.' + code1))
        with open(f2, "w", encoding='utf8') as f:
            for i, wd in enumerate(vocab_list2):
                f.write(wd + " | ")
                f.write(" ".join([str(w) for w in W_y[i]]) + "\n")
        shutil.copyfile(f2, os.path.join(model_path, folder_name + '.' + code2))

    def dump_to_npy(self):
        code1, code2 = self.country_code[self.lang1], self.country_code[self.lang2]
        input_path = os.path.join(self.data_path, self.folder_name)
        f1 = os.path.join(input_path, 'npy', code1)
        f2 = os.path.join(input_path, 'npy', code2)
        pf = os.path.join(input_path, 'npy', 'params.json')
        vocab_f1 = os.path.join(input_path, code1 + '_vocab.json')
        vocab_f2 = os.path.join(input_path, code2 + '_vocab.json')
        with open(vocab_f1, "r", encoding='utf8') as f:
            vocab_table1 = json.load(f)
        with open(vocab_f2, "r", encoding='utf8') as f:
            vocab_table2 = json.load(f)
        print(code1, len(vocab_table1), code2, len(vocab_table2))
        self.config.vocab_size1 = len(vocab_table1)
        self.config.vocab_size2 = len(vocab_table2)

        l1 = []
        for l in self.text1:
            tmp = [self.toks['start']]
            for wd in word_tokenize(l.strip()):
                # 加上小寫限制
                wd = wd.lower()
                if wd in vocab_table1: tmp += [vocab_table1[wd]]
                else: tmp += [self.toks['unk']]
            tmp += [self.toks['stop']]
            l1 += [tmp]

        l2 = []
        for l in self.text2:
            tmp = [self.toks['start']]
            for wd in word_tokenize(l.strip()):
                wd = wd.lower()
                if wd in vocab_table2: tmp += [vocab_table2[wd]]
                else: tmp += [self.toks['unk']]
            tmp += [self.toks['stop']]
            l2 += [tmp]

        assert len(l1) == len(l2)

        # create a(x)
        ll = int(len(l1) / self.config.mini_batch) + 1 \
            if len(l1) % self.config.mini_batch != 0 \
            else int(len(l1) /  self.config.mini_batch)
        a1 = []
        a2 = []
        for i in range(ll):
            t1 = np.zeros(self.config.vocab_size1)
            t2 = np.zeros(self.config.vocab_size2)
            i *= self.config.mini_batch
            for j in l1[i: i + self.config.mini_batch]: t1[j] = 1
            for j in l2[i: i + self.config.mini_batch]: t2[j] = 1
            a1 += [t1]
            a2 += [t2]
        
        with open(f1 + '_csr.pkl', "wb") as f: pickle.dump(sparse.csr_matrix(np.asarray(a1, dtype=np.int32)), f)
        with open(f2 + '_csr.pkl', "wb") as f: pickle.dump(sparse.csr_matrix(np.asarray(a2, dtype=np.int32)), f)
        with open(pf, "w", encoding='utf8') as f:
            params = {'vocab_size1': self.config.vocab_size1, 
                            'vocab_size2': self.config.vocab_size2,
                            'vocab_size': self.config.vocab_size,
                            'dropout': self.config.dropout,
                            'hidden_size': self.config.hidden_size,
                            'mini_batch': self.config.mini_batch,
                            'batch_size': self.config.batch_size,
                            'lr': self.config.lr,
                            'lamda': self.config.lamda,
                            'beta': self.config.beta}
            print(params)
            json.dump(params, f)
        
        
    def load_npy(self):
        code1, code2 = self.country_code[self.lang1], self.country_code[self.lang2]
        in_path = os.path.join(self.data_path, self.folder_name)
        f1 = os.path.join(in_path, 'npy', code1)
        f2 = os.path.join(in_path, 'npy', code2)
        pf = os.path.join(in_path, 'npy', 'params.json')
        with open(pf, "r", encoding='utf8') as f:
            config = json.load(f)
        with open(f1 + '_csr.pkl', "rb") as f:
            a1 = pickle.load(f)
            a1 = a1.todense()
        with open(f2 + '_csr.pkl', "rb") as f:
            a2 = pickle.load(f)
            a2 = a2.todense()
        vocab_f1 = os.path.join(in_path, code1 + '_vocab.json')
        vocab_f2 = os.path.join(in_path, code2 + '_vocab.json')
        with open(vocab_f1, "r", encoding='utf8') as f:
            vocab_table1 = json.load(f)
        with open(vocab_f2, "r", encoding='utf8') as f:
            vocab_table2 = json.load(f)

        return config, (a1, a2), (vocab_table1, vocab_table2)

def run(args):
    prep = Preprocessing(args)
    prep.read_bilingual_data()
    
    # build vocab
    if args.mode == '1':
        prep.build_vocab_table()
    # dump to npy
    elif args.mode == '2':
        prep.dump_to_npy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an BilingualAutoencoder model')
    parser.add_argument('-l1', '--lang1', default='english', help='Language 1.')
    parser.add_argument('-l2', '--lang2', default='spanish', help='Language 2.')
    parser.add_argument('-f', '--folder_name', default='en-zh', help='Folder name.')
    parser.add_argument('-f1', '--file1', default='en-fr.en', help='Language 1.')
    parser.add_argument('-f2', '--file2', default='en-fr.fr', help='Language 2.')
    parser.add_argument('-dp', '--data_path', default='D:/Program/Git/data/', help='Folder name.')
    parser.add_argument('-dn', '--data_name', default='ted_2015', help='Data name.')
    parser.add_argument('-e1', '--embed_name1', default='en-es.en', help='Embedding name 1.')
    parser.add_argument('-e2', '--embed_name2', default='en-es.es', help='Embedding name 2.')
    parser.add_argument('-m', '--mode', default='1', help='Train in what order')
    args = parser.parse_args()

    run(args)

import logging
import pickle
import queue
from os.path import join as pjoin

from util.config import Config

FORMAT = '%(asctime)-15s %(levelname)s:%(message)s'
logger = logging.getLogger("data_loader")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

class DataLoader:
    def __init__(self, src_dir):
        self.config = Config
        self.src_dir = src_dir
        self.load_params(src_dir)
        # self.load_data()
        # self.load_test_data()

    def load_params(self, src_dir):
        logger.info("load params")
        with open(os.path.join(src_dir, "param.plk"), "rb") as f:
            vocab = pickle.load(f)
            vals = pickle.load(f)

        self.vocab = vocab
        self._vocab_size = vals['vocab_size']
        self._max_length = vals['max_length']
        logger.info("vocab size:{}, max_length:{}".format(self._vocab_size, self._max_length))

    def process_dict(self, dir, input_dict):
        vid_list = input_dict['id']
        cap_list = []
        np_list = map(lambda x: pjoin(dir, x + ".npy"), vid_list)

        mask_list = []
        for sent in input_dict['caption']:
            m = [1 for _ in range(len(sent))]
            pad = [0 for _ in range(self.config.cap_length - len(sent))]
            m += pad
            assert len(m) == self.config.cap_length
            mask_list += [m]
            cap_list += [sent + pad]

        input_dict['caption'] = np.asarray(cap_list, dtype=np.int32)
        input_dict['mask'] = np.asarray(mask_list, dtype=np.bool)
        input_dict['img'] = list(np_list)

        return input_dict

    def process_test_dict(self, dir, input_dict):
        vid_list = input_dict['id']
        np_list = map(lambda x: pjoin(dir, x + ".npy"), vid_list)
        input_dict['img'] = list(np_list)

        return input_dict

    def load_data(self):
        logger.info("load training data")
        src_dir = self.src_dir
        # ----- training data -----
        with open(pjoin(src_dir, "train_label.json"), "r") as f:
            train_data = json.load(f)
        train_dir = pjoin("MLDS_hw2_data", "training_data", "feat")
        train_data = self.process_dict(train_dir, train_data)

        # ----- developing data -----
        with open(pjoin(src_dir, "dev_label.json"), "r") as f:
            dev_data = json.load(f)
        dev_data = self.process_dict(train_dir, dev_data)

        self._train_data = train_data
        self._dev_data = dev_data

    def load_test_data(self):
        logger.info("load test data")
        src_dir = self.src_dir

        # ----- testing data -----
        with open(pjoin(src_dir, "test_label.json"), "r") as f:
            test_data = json.load(f)
        test_dir = pjoin("MLDS_hw2_data", "testing_data", "feat")
        test_data = self.process_test_dict(test_dir, test_data)

        self._test_data = test_data

    def data_queue(self, data):
        """

        :param data:
        :return: (img_slice, cap_slice, mask_slice, vid_slice)
        """
        q = queue.Queue()

        vid_list = data['id']
        cap_list = data['caption']
        mask_list = data['mask']
        img_list = data['img']

        assert cap_list.shape[0] == mask_list.shape[0]
        n_samples = cap_list.shape[0]
        idx = np.arange(n_samples)

        start = 0
        end = n_samples
        while start < end:
            cur_end = start + self.config.batch_size
            indices = idx[start:cur_end]

            img_slice = map(lambda x: np.load(img_list[x]), indices)
            vid_slice = map(lambda x: vid_list[x], indices)
            cap_slice = map(lambda x: cap_list[x], indices)
            mask_slice = map(lambda x: mask_list[x], indices)
            q.put((img_slice, cap_slice, mask_slice, vid_slice))

            start = cur_end
        return q

    def test_data_queue(self, data):
        q = queue.Queue()

        vid_list = data['id']
        img_list = data['img']

        n_samples = len(vid_list)
        idx = np.arange(n_samples)

        start = 0
        end = n_samples
        while start < end:
            cur_end = start + self.config.batch_size
            indices = idx[start:cur_end]
            img_slice = map(lambda x: np.load(img_list[x]), indices)
            vid_slice = map(lambda x: vid_list[x], indices)
            q.put((img_slice, vid_slice))

            start = cur_end
        return q

    @property
    def train_queue(self):
        return self.data_queue(self.train_data)

    @property
    def dev_queue(self):
        return self.data_queue(self.dev_data)

    @property
    def test_queue(self):
        return self.test_data_queue(self.test_data)

    @property
    def train_data(self):
        return self._train_data

    @property
    def dev_data(self):
        return self._dev_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_length(self):
        return self._max_length

#  -*- coding: utf-8 -*-
import os
import sys
import json
import unicodecsv
import numpy as np
from tqdm import tqdm

reload(sys)
sys.setdefaultencoding('utf-8')

class Reader:
    def __init__(self, vocab_size=50000):
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

    def load_embed(self, lang1, lang2, embed_name, folder_name):
        code1, code2 = self.country_code[lang1], self.country_code[lang2]
        embed_path = os.path.join(self.embed_path, embed_name, folder_name)

        l1 = folder_name.split('.')[0] + '.' + code1
        l2 = folder_name.split('.')[0] + '.' + code2

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
                sent = sent.split()
                for wd in sent:
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

if __name__ == "__main__":
    pass
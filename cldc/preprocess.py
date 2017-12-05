#  -*- coding: utf-8 -*-
import os
import csv
import sys
import json
import shutil
import jieba
import random
import codecs
import xml.etree.cElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer

reload(sys)
sys.setdefaultencoding('utf-8')

class Preprocessing:
    def __init__(self):
        self.paren_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.data_path = os.path.join(self.paren_path, 'data', 'RCV2_Multilingual_Corpus')
        self.util_path = os.path.join(self.paren_path, 'data', 'utils')
        self.target_topics = set(['CCAT', 'GCAT', 'ECAT', 'MCAT'])
        self.topic_code = {'CCAT':0, 'ECAT':1, 'GCAT':2, 'MCAT':3}

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

    def prepro(self, lang):
        i = 0
        in_path = os.path.join(self.data_path, lang)
        # language
        for l in os.listdir(in_path):
            sub_path = os.path.join(in_path, l)
            # sub director
            for file in os.listdir(sub_path):
                if i % 1000 == 0: print i
                file_path = os.path.join(sub_path, file)
                tree = ET.ElementTree(file=file_path)
                news = {'id': '', 'headline':'', 'content': '', 'topics': []}
                # news info
                for elem in tree.iter(tag='newsitem'):
                    news['id'] = elem.attrib['itemid']
                # news title
                for elem in tree.iter(tag='headline'):
                    news['headline'] = elem.text
                # news content
                for elem in tree.iterfind('text/p'):
                    news['content'] += elem.text
                # news topic
                for elem in tree.iterfind(".//codes[@class='bip:topics:1.0']/code"):
                    news['topics'] += [elem.attrib['code']]

                self.output(lang, news)
                i += 1

    def prepro_rcv1(self):
        i = 0
        in_path = os.path.join(self.paren_path, 'data', 'rcv1')
        for l in os.listdir(in_path):
            sub_path = os.path.join(in_path, l)
            for file in os.listdir(sub_path):
                if i % 1000 == 0: print i
                file_path = os.path.join(sub_path, file)
                tree = ET.ElementTree(file=file_path)
                news = {'id': '', 'content': '', 'topics': []}
                try:
                    # news info
                    for elem in tree.iter(tag='newsitem'):
                        news['id'] = elem.attrib['itemid']
                    # news title
                    for elem in tree.iter(tag='headline'):
                        news['headline'] = elem.text
                    # news content
                    for elem in tree.iterfind('text/p'):
                        news['content'] += elem.text
                    # news topic
                    for elem in tree.iterfind(".//codes[@class='bip:topics:1.0']/code"):
                        news['topics'] += [elem.attrib['code']]

                    self.output('english', news)
                    i += 1
                except AttributeError:
                    print "Attribute error"
                except:
                    print "Unexpected other error"

    def select_single_topic_v2(self, lang, min_len=120):
        in_path = os.path.join(self.paren_path, 'data', 'RCV2_processed')
        out_path = os.path.join(self.paren_path, 'data', 'RCV2_processed_v2')

        # language
        # for l in os.listdir(path):
        i = 0
        sub_path = os.path.join(in_path, lang)
        sub_outpath = os.path.join(out_path, lang)
        if not os.path.exists(sub_outpath): os.mkdir(sub_outpath)
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()  # news_id = lines[0].strip()
                topics = set(lines[1].strip().split(','))
                inter_topics = self.target_topics.intersection(topics)
                # content = lines[2].strip() + " " + lines[3].strip()  # put file to the assigned topic folder
                subsub_outpath = os.path.join(sub_outpath, inter_topics.pop())
                if not os.path.exists(subsub_outpath): os.mkdir(subsub_outpath)

                shutil.copyfile(file_path, os.path.join(subsub_outpath, file))
                i += 1
                if i % 1000 == 0: print i

        print lang, i

    def select_single_topic(self, lang, min_len=120):
        in_path = os.path.join(self.paren_path, 'data', 'RCV2')
        out_path = os.path.join(self.paren_path, 'data', 'RCV2_processed')

        # language
        # for l in os.listdir(path):
        i = 0
        sub_path = os.path.join(in_path, lang)
        sub_outpath = os.path.join(out_path, lang)
        if not os.path.exists(sub_outpath): os.mkdir(sub_outpath)
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            with open(file_path, "r") as f:
                try:
                    lines = f.readlines()  # news_id = lines[0].strip()
                    topics = set(lines[1].strip().split(','))
                    inter_topics = self.target_topics.intersection(topics)
                    content = lines[2].strip() + " " + lines[3].strip()
                    if len(inter_topics) == 1 and len(content) > min_len:
                        # put file to the assigned topic folder
                        subsub_outpath = os.path.join(sub_outpath, inter_topics.pop())
                        if not os.path.exists(subsub_outpath): os.mkdir(subsub_outpath)

                        shutil.copyfile(file_path, os.path.join(subsub_outpath, file))
                        i += 1
                        if i % 1000 == 0: print i
                except IndexError:
                    print "Index error"
                except:
                    print "Other error"

        print lang, i

    def cut_chinese(self):
        in_path = os.path.join(self.paren_path, 'data', 'RCV2', 'chinese')
        out_path = os.path.join(self.paren_path, 'data', 'RCV2', 'chinese')

        for file in os.listdir(in_path):
            in_file = os.path.join(in_path, file)
            out_file = os.path.join(out_path, file)
            with open(in_file, "r") as f:
                lines = f.readlines()
                # headline
                lines[2] = " ".join(jieba.cut(lines[2].strip(), cut_all=False)) + "\n"
                # content
                lines[3] = " ".join(jieba.cut(lines[3].strip(), cut_all=False)) + "\n"

            with open(out_file, "w") as f:
                for l in lines:
                    f.write(l)

    def cal_idf(self, lang):
        in_path = os.path.join(self.paren_path, 'data', 'RCV2_processed', lang)
        out_path = os.path.join(self.util_path, lang, 'idf.json')

        vectorizer = TfidfVectorizer(use_idf=True, decode_error='ignore', smooth_idf=False)
        document = []
        i = 0
        for file in os.listdir(in_path):
            file_path = os.path.join(in_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
                # news_id = lines[0].strip()
                # topics = lines[1].strip().split(',')
                content = lines[2].strip()
                content += ' ' + lines[3].strip()
                i += 1
                if i % 1000 == 0: print i
                document += [content]

        vectorizer.fit(document)
        with open(out_path, "w") as f:
            json.dump(dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)), f, encoding='utf8')

    def split_training_valid_test(self, lang, max_num=20000, train_split_ratio=0.6, test_split_ratio=0.2):
        in_path = os.path.join(self.paren_path, 'data', 'RCV2_processed', lang)
        out_path = os.path.join(self.paren_path, 'data', 'utils', lang)
        if not os.path.exists(out_path): os.mkdir(out_path)

        file_list = os.listdir(in_path)
        random.shuffle(file_list)
        file_list = file_list[:max_num]
        n_training = int(len(file_list) * train_split_ratio)
        n_testing = int(len(file_list) * test_split_ratio)
        training_file, testing_file, valid_file = file_list[:n_training], file_list[n_training:n_training+n_testing], file_list[n_training+n_testing:]

        with open(os.path.join(out_path, 'training_file.txt'), "w") as f:
            json.dump(training_file, f)
        with open(os.path.join(out_path, 'testing_file.txt'), "w") as f:
            json.dump(testing_file, f)
        with open(os.path.join(out_path, 'valid_file.txt'), "w") as f:
            json.dump(valid_file, f)

    def read_util_file(self, lang):
        upath = os.path.join(self.util_path, lang)
        with open(os.path.join(upath, 'training_file.txt'), "r") as f:
            training_files = json.load(f)
        with open(os.path.join(upath, 'testing_file.txt'), "r") as f:
            testing_files = json.load(f)
        with open(os.path.join(upath, 'valid_file.txt'), "r") as f:
            valid_files = json.load(f)
        with open(os.path.join(upath, lang + '_idf.json'), "r") as f:
            idf_dict = json.load(f)
        return training_files, testing_files, valid_files, idf_dict

    def merge(self, lang):
        in_path = os.path.join(self.paren_path, 'data', 'RCV2_processed', lang)
        out_path = os.path.join(self.paren_path, 'data', 'RCV2_merged', lang)

        if not os.path.exists(out_path): os.mkdir(out_path)

        def out_merge(in_path, out_path, file_names):
            header = ['doc_id', 'class', 'headline', 'content']
            with open(out_path, "wb") as csvfile:
                csvfile.write(codecs.BOM_UTF8)
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(header)
                for file_name in file_names:
                    with open(os.path.join(in_path, file_name), "r") as f:
                        lines = f.readlines()
                        doc_id = lines[0].strip()
                        topics = self.target_topics.intersection(set(lines[1].strip().split(',')))
                        headline = lines[2].strip()
                        content = lines[3].strip()

                        csv_writer.writerow([doc_id, self.topic_code[topics.pop()], headline, content])

        training_files, testing_files, valid_files, _ = self.read_util_file(lang)
        out_merge(in_path, os.path.join(out_path, 'train.csv'), training_files)
        out_merge(in_path, os.path.join(out_path, 'test.csv'), testing_files)
        out_merge(in_path, os.path.join(out_path, 'valid.csv'), valid_files)


if __name__ == "__main__":
    prep = Preprocessing()
    lang = ['chinese', 'english', 'french', 'german', 'italian', 'spanish']
    #prep.prepro_rcv1()
    #prep.cut_chinese()
    for i in lang:
        print i
        prep.select_single_topic_v2(i)
        # prep.split_training_valid_test(i)
        # prep.cal_idf(i)
        # prep.merge(i)
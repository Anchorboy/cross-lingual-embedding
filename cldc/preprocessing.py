#  -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
import jieba
import xml.etree.cElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer

reload(sys)
sys.setdefaultencoding('utf-8')


class Preprocessing:
    def __init__(self):
        self.paren_path = os.path.abspath('..')
        self.path = os.path.join(self.paren_path, 'data', 'RCV2_Multilingual_Corpus')

    def output(self, dir_name, news):
        path = os.path.join(self.paren_path, 'data', 'RCV2')
        if not os.path.exists(path): os.mkdir(path)
        sub_path = os.path.join(path, dir_name)
        if not os.path.exists(sub_path): os.mkdir(sub_path)

        file_path = os.path.join(sub_path, news['id'] + '.txt')
        with open(file_path, "w") as f:
            # json.dump(news, f)
            f.write(news['id'] + "\n")
            f.write(",".join(news['topics']) + "\n")
            f.write(news['content'].decode('utf8') + "\n")

    def prepro(self, dir_name):
        i = 0
        path = os.path.join(self.path, dir_name)
        # language
        for l in os.listdir(path):
            sub_path = os.path.join(path, l)
            # sub director
            for file in os.listdir(sub_path):
                if i % 1000 == 0: print i
                file_path = os.path.join(sub_path, file)
                tree = ET.ElementTree(file=file_path)
                news = {'id': '', 'content': '', 'topics': []}
                # news info
                for elem in tree.iter(tag='newsitem'):
                    news['id'] = elem.attrib['itemid']
                # news title
                for elem in tree.iterfind('text/p'):
                    news['content'] += elem.text
                # news topic
                for elem in tree.iterfind(".//codes[@class='bip:topics:1.0']/code"):
                    news['topics'] += [elem.attrib['code']]

                self.output(dir_name, news)
                i += 1

    def prepro_rcv1(self):
        i = 0
        path = os.path.join(self.paren_path, 'data', 'rcv1')
        for l in os.listdir(path):
            sub_path = os.path.join(path, l)
            for file in os.listdir(sub_path):
                if i % 1000 == 0: print i
                file_path = os.path.join(sub_path, file)
                tree = ET.ElementTree(file=file_path)
                news = {'id': '', 'content': '', 'topics': []}
                # news info
                for elem in tree.iter(tag='newsitem'):
                    news['id'] = elem.attrib['itemid']
                # news title
                for elem in tree.iterfind('text/p'):
                    news['content'] += elem.text
                # news topic
                for elem in tree.iterfind(".//codes[@class='bip:topics:1.0']/code"):
                    news['topics'] += [elem.attrib['code']]

                self.output('english', news)
                i += 1

    def select_single_topic(self, lang):
        path = os.path.join(self.paren_path, 'data', 'RCV2')
        outpath = os.path.join(self.paren_path, 'data', 'RCV2_processed')

        target_topics = set(['CCAT', 'GCAT', 'ECAT', 'MCAT'])
        # language
        # for l in os.listdir(path):
        i = 0
        sub_path = os.path.join(path, lang)
        sub_outpath = os.path.join(outpath, lang)
        if not os.path.exists(sub_outpath): os.mkdir(sub_outpath)
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
                # news_id = lines[0].strip()
                topics = set(lines[1].strip().split(','))
                inter_topics = target_topics.intersection(topics)
                # content = lines[2].strip()
                if len(inter_topics) == 1:
                    shutil.copyfile(file_path, os.path.join(sub_outpath, file))
                    i += 1
                    if i % 1000 == 0: print i

        print lang, i

    def cut_chinese(self):
        in_path = os.path.join(self.paren_path, 'data', 'RCV2', 'chinese')
        out_path = os.path.join(self.paren_path, 'data', 'RCV2', 'chinese_cut')

        for file in os.listdir(in_path):
            in_file = os.path.join(in_path, file)
            out_file = os.path.join(out_path, file)
            with open(in_file, "r") as f:
                lines = f.readlines()
                lines[2] = " ".join(jieba.cut(lines[2].strip(), cut_all=False)) + "\n"

            with open(out_file, "w") as f:
                for l in lines:
                    f.write(l)

    def test(self):
        path = os.path.join(self.paren_path, 'data', 'RCV2_processed')
        # outpath = os.path.join(self.paren_path, 'data', 'RCV2_processed')

        # language
        l = 'english'

        count = 0
        print l
        sub_path = os.path.join(path, l)
        topic_set = set(['CCAT', 'ECAT', 'GCAT', 'MCAT'])
        # sub_outpath = os.path.join(outpath, l)
        # if not os.path.exists(sub_outpath): os.mkdir(sub_outpath)
        cnt = {}
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
                # news_id = lines[0].strip()
                topics = set(lines[1].strip().split(','))
                # content = lines[2].strip()
                inter_s = topic_set.intersection(topics)
                if len(inter_s) == 1:
                    for i in inter_s:
                        if i not in cnt:
                            cnt[i] = 1
                        else:
                            cnt[i] += 1
                    count += 1
                    if count % 1000 == 0:
                        print count, cnt

        print topic_set

    def cal_idf(self, lang):
        path = os.path.join(self.paren_path, 'data', 'RCV2', lang)
        outpath = os.path.join(self.paren_path, 'data', 'embedding', 'idf', lang + '_idf.json')

        vectorizer = TfidfVectorizer(use_idf=True, decode_error='ignore', smooth_idf=False)
        document = []
        i = 0
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
                # news_id = lines[0].strip()
                # topics = lines[1].strip().split(',')
                content = lines[2].strip()
                i += 1
                if i % 1000 == 0: print i
                document += [content]

        vectorizer.fit(document)
        with open(outpath, "w") as f:
            json.dump(dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)), f)


if __name__ == "__main__":
    prep = Preprocessing()
    lang = ['chinese_cut', 'english', 'french', 'german', 'italian']
    # for i in lang:
    #     print i
    # prep.prepro('spanish')
    # prep.prepro_rcv1()
    # for i in lang:
        # print i
        # prep.select_single_topic(i)
    prep.cal_idf('spanish')
    # for i in lang:
    #     print i
    #     prep.cal_idf(i)


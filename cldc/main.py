import os
import csv
import sys
import logging
import argparse
from reader import Reader
from model import AveragedPerceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

# create a logger
logger = logging.getLogger('util_logger')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fh = logging.FileHandler('test.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

def pred(mode, lang1, lang2, X1_data, X2_data, sampled=1000):
    train_X1, train_y1, test_X1, test_y1, valid_X1, valid_y1 = X1_data
    train_X2, train_y2, test_X2, test_y2, valid_X2, valid_y2 = X2_data
    clf = AveragedPerceptron(max_iter=15)
    logger.info('training classifier')
    if mode == '1':
        clf.fit(train_X1, train_y1, valid_X1, valid_y1)
    elif mode == '2':
        clf.fit(train_X2, train_y2, valid_X2, valid_y2)
    elif mode == '12':
        clf.fit(train_X1, train_y1, valid_X1, valid_y1)
        clf.fit(train_X2, train_y2, valid_X2, valid_y2)
    elif mode == '21':
        clf.fit(train_X2, train_y2, valid_X2, valid_y2)
        clf.fit(train_X1, train_y1, valid_X1, valid_y1)
    # logger.info('training classifier on ' + lang1)
    # clf.fit(train_X1, train_y1)
    # clf.save_weight(lang=lang1, embed_name='I_matrix')

    accs = [mode]
    logger.info('evaluate on ' + lang1 + ' training set')
    accs += [clf.evaluate(train_X1[:sampled], train_y1[:sampled])]
    accs += [clf.evaluate(train_X1, train_y1)]

    logger.info('evaluate on ' + lang1 + ' test set')
    accs += [clf.evaluate(test_X1[:sampled], test_y1[:sampled])]
    accs += [clf.evaluate(test_X1, test_y1)]

    logger.info('evaluate on ' + lang2 + ' training set')
    accs += [clf.evaluate(train_X2[:sampled], train_y2[:sampled])]
    accs += [clf.evaluate(train_X2, train_y2)]

    logger.info('evaluate on ' + lang2 + ' test set')
    accs += [clf.evaluate(test_X2[:sampled], test_y2[:sampled])]
    accs += [clf.evaluate(test_X2, test_y2)]

    return accs

def read_files(args):
    lang1 = args.lang1
    lang2 = args.lang2
    embed_name1 = args.embed_name1
    embed_name2 = args.embed_name2
    folder_name = args.folder_name
    mode = args.mode
    pos_modes = ['1', '2', '12', '21', 'all']
    assert mode in pos_modes

    # --- init reader
    reader = Reader()

    # --- read files for language 1
    logger.info('read ' + lang1)
    X1_data = reader.read_files(lang1)
    train_X1, train_y1, test_X1, test_y1, valid_X1, valid_y1 = X1_data
    idf1 = reader.read_idf(lang1)

    # --- read files for language 2
    logger.info('read ' + lang2)
    X2_data = reader.read_files(lang2)
    train_X2, train_y2, test_X2, test_y2, valid_X2, valid_y2 = X2_data
    idf2 = reader.read_idf(lang2)

    # --- read embedding
    logger.info('read embedding')
    word_dict1, embed_mat1, word_dict2, embed_mat2 = reader.load_embed(embed_name1, embed_name2, folder_name)

    # --- vectorize language 1
    logger.info('vectorize ' + lang1)
    train_X1, test_X1, valid_X1 = reader.vectorize([train_X1, test_X1, valid_X1], idf1, word_dict1, embed_mat1)

    # --- vectorize language 2
    logger.info('vectorize ' + lang2)
    train_X2, test_X2, valid_X2 = reader.vectorize([train_X2, test_X2, valid_X2], idf2, word_dict2, embed_mat2)

    X1_data = (train_X1, train_y1, test_X1, test_y1, valid_X1, valid_y1)
    X2_data = (train_X2, train_y2, test_X2, test_y2, valid_X2, valid_y2)
    return X1_data, X2_data

def write_to_csv(rows, file_name):
    header = ['mode', 'train1 sampled', 'train1', 'test1 sampled', 'test1', 'train2 sampled', 'train2', 'test2 sampled',
              'test2']
    logger.info('write to csv file')
    with open(os.path.join('results', file_name), "wb") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

def run(args):
    lang1 = args.lang1
    lang2 = args.lang2
    mode = args.mode
    pos_modes = ['1', '2', '12', '21', 'all']
    assert mode in pos_modes

    X1_data, X2_data = read_files(args)

    rows = []
    # --- train classifier model
    if mode == 'all':
        for m in pos_modes[:-1]:
            rows += [pred(m, lang1, lang2, X1_data, X2_data)]
    else:
        rows += [pred(mode, lang1, lang2, X1_data, X2_data)]

    write_to_csv(rows, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an BilingualAutoencoder model')
    parser.add_argument('-l1', '--lang1', default='english', help='Language 1.')
    parser.add_argument('-l2', '--lang2', default='spanish', help='Language 2.')
    parser.add_argument('-f', '--folder_name', default='en-zh', help='Folder name.')
    parser.add_argument('-e1', '--embed_name1', default='en-es.en', help='Embedding name 1.')
    parser.add_argument('-e2', '--embed_name2', default='en-es.es', help='Embedding name 2.')
    parser.add_argument('-m', '--mode', default='all', help='Train in what order')
    parser.add_argument('-o', '--output_path', default='en-es', help='Train in what order')
    args = parser.parse_args()

    if not os.path.exists('results'): os.mkdir('results')
    lang = ['chinese', 'english', 'french', 'german', 'italian', 'spanish']
    run(args)
    # run(lang1='english', lang2='french', embed_name='BAE', folder_name='en-fr')
    # run(lang1='english', lang2='spanish', embed_name='BAE', folder_name='en-es')
    # run(lang1='english', lang2='german', embed_name='I-Matrix', folder_name='de-en.40')
    # run_I_matrix('english', 'french', 'fr-en.80')
    # run_I_matrix('spanish', 'english', 'es-en.80')
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import time

import tensorflow as tf

from model import BilingualAutoencoder
from util.config import Config
from util.preprocess import load_npy

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
logger = logging.getLogger("rnn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def do_test(args):
    logger.info("Training rnn model")
    config = Config(args)
    args.mode = 'train'

    logger.info("loading")
    loaded_config, a1, a2, vocab1, vocab2 = load_npy(args)
    # data = (a1, a2)
    config.vocab_size = loaded_config['vocab_size']
    config.vocab_size1 = loaded_config['vocab_size1']
    config.vocab_size2 = loaded_config['vocab_size2']
    config.hidden_size = int(args.dim)

    logger.info("building")
    # train_data, train_labels, train_mask
    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = BilingualAutoencoder(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=tf_config) as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            # model.fit(session, saver, data, logger)
            model.save_weight(session)

def do_train(args):
    logger.info("Training rnn model")
    config = Config(args)
    args.mode = "train"

    logger.info("loading")
    loaded_config, data, vocab = load_npy(args)
    config.vocab_size = loaded_config['vocab_size']
    config.vocab_size1 = loaded_config['vocab_size1']
    config.vocab_size2 = loaded_config['vocab_size2']
    config.hidden_size = int(args.dim)
    
    logger.info("building")
    # train_data, train_labels, train_mask
    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = BilingualAutoencoder(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=tf_config) as session:
            session.run(init)
            model.fit(session, saver, data, vocab, logger)

def main():
    parser = argparse.ArgumentParser(description='Trains and tests an BilingualAutoencoder model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.add_argument('-m', '--model_path', help="Model path.")
    command_parser.add_argument('-d', '--dim', default='256', help='Dimension of generated embedding.')
    command_parser.add_argument('-l1', '--lang1', default='english', help='Language 1.')
    command_parser.add_argument('-l2', '--lang2', default='chinese', help='Language 2.')
    command_parser.add_argument('-f1', '--file1', default='en-fr.en', help='Language 2.')
    command_parser.add_argument('-f2', '--file2', default='en-fr.fr', help='Language 2.')
    command_parser.add_argument('-fn', '--folder_name', default='en-zh', help='Folder name.')
    command_parser.add_argument('-dp', '--data_path', default='D:/Program/Git/data/', help='Folder name.')
    command_parser.add_argument('-dn', '--data_name', default='ted_2015', help='Data name.')
    command_parser.set_defaults(func=do_test)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-m', '--model_path', help="Model path.")
    command_parser.add_argument('-d', '--dim', default='256', help='Dimension of generated embedding.')
    command_parser.add_argument('-l1', '--lang1', default='english', help='Language 1.')
    command_parser.add_argument('-l2', '--lang2', default='chinese', help='Language 2.')
    command_parser.add_argument('-f1', '--file1', default='en-fr.en', help='Language 2.')
    command_parser.add_argument('-f2', '--file2', default='en-fr.fr', help='Language 2.')
    command_parser.add_argument('-fn', '--folder_name', default='en-zh', help='Folder name.')
    command_parser.add_argument('-dp', '--data_path', default='D:/Program/Git/data/', help='Data path.')
    command_parser.add_argument('-dn', '--data_name', default='ted_2015', help='Data name.')
    command_parser.set_defaults(func=do_train)

    # command_parser = subparsers.add_parser('evaluate', help='')
    # command_parser.add_argument('-m', '--model-path', help="Training data")
    # command_parser.add_argument('-d', '--dim', default='200', help='Dimension of generated embedding.')
    # command_parser.add_argument('-l1', '--lang1', default='en', help='Language 1.')
    # command_parser.add_argument('-l2', '--lang2', default='zh', help='Language 2.')
    # command_parser.add_argument('-f', '--folder_name', default='en-zh', help='Folder name.')
    # command_parser.set_defaults(func=do_evaluate)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
import pandas as pd
import tensorflow as tf
from util.config import Config
from util.preprocess import load_npy
from model import BilingualAutoencoder

logger = logging.getLogger("rnn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# def do_test(args):
#     logger.info("Testing rnn model")
#     args.mode = "train"
#     config = Config(args)
# 
#     logger.info('loading')
#     data_loader = DataLoader("test")
#     data_loader.load_and_preprocess()
#     
#     config.max_length = data_loader.max_length
#     config.vocab_size = data_loader.vocab_size
# 
#     logger.info('building')
#     # train_data, train_labels, train_mask
#     with tf.Graph().as_default():
#         logger.info("Building model...",)
#         start = time.time()
#         model = RNNModel(config, pretrained_embeddings)
#         logger.info("took %.2f seconds", time.time() - start)
# 
#         init = tf.global_variables_initializer()
#         saver = tf.train.Saver()
# 
#         with tf.Session() as session:
#             session.run(init)
#             model.fit(session, saver, data_loader, logger)

def do_train(args):
    logger.info("Training rnn model")
    config = Config(args)
    args.mode = "train"

    logger.info("loading")
    loaded_config, a1, a2 = load_npy(lang1=args.lang1, lang2=args.lang2, folder_name=args.folder_name)
    data = (a1, a2)
    config.vocab_size = loaded_config.vocab_size
    config.hidden_size = args.dim
    
    logger.info("building")
    # train_data, train_labels, train_mask
    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = BilingualAutoencoder(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, data, logger)

def main():
    parser = argparse.ArgumentParser(description='Trains and tests an BilingualAutoencoder model')
    subparsers = parser.add_subparsers()

    # command_parser = subparsers.add_parser('test', help='')
    # command_parser.add_argument('-d', '--dim', default='200', help='Dimension of generated embedding.')
    # command_parser.add_argument('-l1', '--lang1', default='en', help='Language 1.')
    # command_parser.add_argument('-l2', '--lang2', default='zh', help='Language 2.')
    # command_parser.add_argument('-f', '--folder_name', default='en-zh', help='Folder name.')
    # command_parser.set_defaults(func=do_test)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-d', '--dim', default='256', help='Dimension of generated embedding.')
    command_parser.add_argument('-l1', '--lang1', default='english', help='Language 1.')
    command_parser.add_argument('-l2', '--lang2', default='chinese', help='Language 2.')
    command_parser.add_argument('-f', '--folder_name', default='en-zh', help='Folder name.')
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

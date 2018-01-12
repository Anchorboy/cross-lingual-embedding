# -*- coding: utf-8 -*-
import os
import sys
import math
# parendir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.insert(0, parendir)
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf
from util.preprocess import dump_word_embedding

class BilingualAutoencoder:
    def __init__(self, config):
        self.config = config

        self.input_placeholder1 = None
        self.input_placeholder2 = None
        self.labels_placeholder1 = None
        self.labels_placeholder2 = None
        self.dropout_placeholder = None

        self.tensor_dict = {}
        self.array_dict = {}
        self.params = []
        self.build()

        # --- model architecture ---

    def add_placeholders(self):
        """
        總共有幾個placeholders
        input_placeholder: (None, time_steps, )
        mask_placeholder
        labels_placeholder
        dropout_placeholder

        :return:
        """

        self.input_placeholder1 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size1))
        self.input_placeholder2 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size2))
        self.labels_placeholder1 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size1))
        self.labels_placeholder2 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size2))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(self, inputs_batch1, inputs_batch2, labels_batch1=None, labels_batch2=None, dropout=1):
        feed_dict = {self.input_placeholder1: inputs_batch1,
                     self.input_placeholder2: inputs_batch2,
                     self.dropout_placeholder: dropout}
        if labels_batch1 is not None and labels_batch2 is not None:
            feed_dict[self.labels_placeholder1] = labels_batch1
            feed_dict[self.labels_placeholder2] = labels_batch2
        return feed_dict

    def add_prediction_op(self):
        x = self.input_placeholder1
        y = self.input_placeholder2
        xy = tf.concat([x, y], axis=1)
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("encoder"):
            W = tf.get_variable('W', shape=(self.config.vocab_size1 + self.config.vocab_size2, self.config.hidden_size),
                                  initializer=tf.random_uniform_initializer(
                                      minval=-4 * np.sqrt(6. / self.config.hidden_size),
                                      maxval=4 * np.sqrt(6. / self.config.hidden_size)))
            W_x = tf.slice(W, [0, 0], [self.config.vocab_size1, self.config.hidden_size])
            W_y = tf.slice(W, [self.config.vocab_size1, 0], [self.config.vocab_size2, self.config.hidden_size])
            b = tf.get_variable('b', shape=(self.config.hidden_size), initializer=tf.contrib.layers.xavier_initializer(seed=123))

        with tf.variable_scope("decoder"):
            W_t = tf.transpose(W)
            W_xt = tf.transpose(W_x)
            W_yt = tf.transpose(W_y)
            b_t = tf.get_variable('b_t', shape=(self.config.vocab_size1 + self.config.vocab_size2), initializer=tf.contrib.layers.xavier_initializer(seed=125))
            b_xt = tf.slice(b_t, [0], [self.config.vocab_size1])
            b_yt = tf.slice(b_t, [self.config.vocab_size1], [self.config.vocab_size2])

        # a_x
        a_x = tf.nn.xw_plus_b(x, W_x, b)
        a_y = tf.nn.xw_plus_b(y, W_y, b)
        a_xy = tf.nn.xw_plus_b(xy, W, b)
        # h(a(x)) = hyperbolic of a(x)
        h_a_x = tf.nn.sigmoid(a_x)
        h_a_x_dropout = tf.nn.dropout(h_a_x, keep_prob=dropout_rate)
        h_a_y = tf.nn.sigmoid(a_y)
        h_a_y_dropout = tf.nn.dropout(h_a_y, keep_prob=dropout_rate)
        h_a_xy = tf.nn.sigmoid(a_xy)
        h_a_xy_dropout = tf.nn.dropout(h_a_xy, keep_prob=dropout_rate)

        # z1 = f(x, y), z2 = f(y, x), z3 = f(x), z4 = f(y), z5 = f((x,y), (x,y)), z6 = cor(a(x), a(y))
        z1 = tf.nn.xw_plus_b(h_a_x_dropout, W_yt, b_yt)
        z2 = tf.nn.xw_plus_b(h_a_y_dropout, W_xt, b_xt)
        # z3 = tf.nn.xw_plus_b(h_a_x_dropout, W_xt, b_xt)
        z3 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_x_dropout, W_xt, b_xt))
        # z4 = tf.nn.xw_plus_b(h_a_y_dropout, W_yt, b_yt)
        z4 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_y_dropout, W_yt, b_yt))
        z5 = tf.nn.xw_plus_b(h_a_xy_dropout, W_t, b_t)
        
        # cor = []
        x1 = a_x - tf.tile(tf.reduce_mean(a_x, axis=0, keep_dims=True), (tf.shape(a_x)[0], 1))
        x2 = a_y - tf.tile(tf.reduce_mean(a_x, axis=0, keep_dims=True), (tf.shape(a_y)[0], 1))
        nr = tf.reduce_sum(tf.multiply(x1, x2), axis=0) / (tf.sqrt(tf.reduce_sum(tf.square(x1), axis=0)) * tf.sqrt(tf.reduce_sum(tf.square(x2), axis=0)))
        cor = -nr
        # for i in range(0, self.config.hidden_size):
        #     x1 = a_x[:, i] - (tf.multiply(tf.ones(self.config.batch_size), (tf.reduce_sum(a_x[:, i]) / self.config.batch_size)))
        #     x2 = a_y[:, i] - (tf.multiply(tf.ones(self.config.batch_size), (tf.reduce_sum(a_y[:, i]) / self.config.batch_size)))
        #     nr = tf.reduce_sum(tf.multiply(x1, x2)) / (tf.sqrt(tf.reduce_sum(tf.square(x1))) * tf.sqrt(tf.reduce_sum(tf.square(x2))))
        #     cor += [-nr]

        z6 = cor

        preds = [z1, z2, z3, z4, z5, z6]
        for pred in [z2, z3]:
            assert pred.get_shape().as_list() == [None, self.config.vocab_size1], \
                "predictions are not of the right shape. Expected {}, got {}".format(
                    [None, self.config.vocab_size1], pred.get_shape().as_list())
        for pred in [z1, z4]:
            assert pred.get_shape().as_list() == [None, self.config.vocab_size2], \
                "predictions are not of the right shape. Expected {}, got {}".format(
                    [None, self.config.vocab_size2], pred.get_shape().as_list())
        self.tensor_dict['W'] = W
        self.tensor_dict['W_x'] = W_x
        self.tensor_dict['W_y'] = W_y
        self.params = [W, b, b_t]
        return preds

    def add_loss_op(self, preds):
        # preds = [(x, y), (y, x), (x), (y), ((x,y), (x, y)), cor(a(x), a(y))]
        x_labels = self.labels_placeholder1
        y_labels = self.labels_placeholder2
        xy_labels = tf.concat([x_labels, y_labels], axis=1)
        l1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[0], labels=y_labels), axis=1)
        l2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[1], labels=x_labels), axis=1)
        # l3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[2], labels=x_labels), axis=1)
        l3 = tf.reduce_sum(tf.square(x_labels - preds[2]) / 2, axis=1) 
        # l4 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[3], labels=y_labels), axis=1)
        l4 = tf.reduce_sum(tf.square(y_labels - preds[3]) / 2, axis=1) 
        l5 = tf.reduce_sum(self.config.beta * tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[4], labels=xy_labels), axis=1)
        l6 = self.config.lamda * tf.reduce_sum(preds[5])
        L1 = tf.reduce_mean(l1 + l2 + l5 + l6 + 200) + self.config.rr * tf.nn.l2_loss(self.tensor_dict['W'])
        L2 = tf.reduce_mean(l3) + self.config.rr * tf.nn.l2_loss(self.tensor_dict['W_x'])
        L3 = tf.reduce_mean(l4) + self.config.rr * tf.nn.l2_loss(self.tensor_dict['W_y'])
        L = [L1, L2, L3]
        l_ = [tf.reduce_mean(l1), tf.reduce_mean(l2), tf.reduce_mean(l3), tf.reduce_mean(l4), tf.reduce_mean(l5), tf.reduce_mean(l6), L1, L2, L3]
        return L, l_

    def add_cost_op(self, loss):
        cost = tf.reduce_mean(loss)
        return cost

    def add_training_op(self, cost):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        grads_and_vars = optimizer.compute_gradients(cost, self.params)
        train_op = optimizer.apply_gradients(grads_and_vars)
        return train_op

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss, self.loss_ = self.add_loss_op(self.pred)
        self.cost = self.add_cost_op(self.loss)
        self.train_ops = []
        for L in self.loss:
            self.train_ops += [self.add_training_op(L)]
        self.sum_op = self.add_summary_op()

    def add_summary_op(self):
        # preds = [(x, y), (y, x), (x), (y), ((x,y), (x, y)), cor(a(x), a(y))]
        loss = self.loss_
        tf.summary.scalar('l_x_y', loss[0])
        tf.summary.scalar('l_y_x', loss[1])
        tf.summary.scalar('l_x', loss[2])
        tf.summary.scalar('l_y', loss[3])
        tf.summary.scalar('l_x_y_x_y', loss[4])
        tf.summary.scalar('cor_a_x_a_y', loss[5])
        tf.summary.scalar('L1', loss[6])
        tf.summary.scalar('L2', loss[7])
        tf.summary.scalar('L3', loss[8])
        tf.summary.scalar('cost', self.cost)
        merged = tf.summary.merge_all()
        return merged

    # --- model operation ---

    def train_on_batch(self, sess, inputs_batch1, inputs_batch2, labels_batch1, labels_batch2):
        feed = self.create_feed_dict(inputs_batch1=inputs_batch1, inputs_batch2=inputs_batch2, labels_batch1=labels_batch1, labels_batch2=labels_batch2, dropout=self.config.dropout)
        for train_op in self.train_ops:
            sess.run([train_op], feed_dict=feed)
        cost, summary = sess.run([self.cost, self.sum_op], feed_dict=feed)
        return cost, summary

    def run_epoch(self, sess, data, logger, writer, rand=True):
        x_data, y_data = data
        x_shp = x_data.shape
        y_shp = y_data.shape
        assert x_shp[0] == y_shp[0]

        logger.info('start epoch')
        n_batch = int(math.ceil(x_shp[0] / self.config.batch_size))
        pbar = trange(n_batch)
        for i in pbar:
            if rand:
                rand_idx = np.random.permutation(x_shp[0])
                x_batch = x_data[rand_idx[i: i+self.config.batch_size]]
                y_batch = y_data[rand_idx[i: i+self.config.batch_size]]
            else:
                x_batch = x_data[i: i + self.config.batch_size]
                y_batch = y_data[i: i + self.config.batch_size]
            batch = (x_batch, y_batch, x_batch, y_batch)
            cost, summary = self.train_on_batch(sess, *batch)

            writer.add_summary(summary, i)
            pbar.set_description("train loss = {:.2f}".format(cost))
        print("")

    def fit(self, sess, saver, data, vocab, logger):
        writer = tf.summary.FileWriter(self.config.log_output + '/train',
                                             sess.graph)
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            self.run_epoch(sess, data, logger, writer)
            if saver:
                saver.save(sess, self.config.model_output)
            if epoch % 10 == 0 and epoch > 0:
                logger.info('dump word embedding')
                self.array_dict['W_x'] = sess.run(self.tensor_dict['W_x'])
                self.array_dict['W_y'] = sess.run(self.tensor_dict['W_y'])
                dump_word_embedding(self.array_dict['W_x'], self.array_dict['W_y'], vocab[0], vocab[1],
                                    self.config.output_path, self.config.lang1, self.config.lang2)
            print("")
        logger.info('dump word embedding')
        self.array_dict['W_x'] = sess.run(self.tensor_dict['W_x'])
        self.array_dict['W_y'] = sess.run(self.tensor_dict['W_y'])
        dump_word_embedding(self.array_dict['W_x'], self.array_dict['W_y'], vocab[0], vocab[1], self.config.output_path, self.config.lang1, self.config.lang2)

    def save_weight(self, sess):
        self.array_dict['W_x'] = sess.run(self.tensor_dict['W_x'])
        self.array_dict['W_y'] = sess.run(self.tensor_dict['W_y'])
        np.save(self.config.output_path + 'W_x.npy', self.array_dict['W_x'])
        np.save(self.config.output_path + 'W_y.npy', self.array_dict['W_y'])

def run():
    # config = Config()
    pass


if __name__ == "__main__":
    # model = BilingualAutoencoder()
    print("hello")

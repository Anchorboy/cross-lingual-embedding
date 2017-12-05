# -*- coding: utf-8 -*-
import os
import sys
import math
parendir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parendir)
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf


class BilingualAutoencoder:
    def __init__(self, config):
        self.config = config

        self.input_placeholder1 = None
        self.input_placeholder2 = None
        self.labels_placeholder1 = None
        self.labels_placeholder2 = None
        self.dropout_placeholder = None

        self.tensor_dict = {}
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

        self.input_placeholder1 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size))
        self.input_placeholder2 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size))
        self.labels_placeholder1 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size))
        self.labels_placeholder2 = tf.placeholder(dtype=tf.float32, shape=(None, self.config.vocab_size))
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
            W = tf.get_variable('W', shape=(2 * self.config.vocab_size, self.config.hidden_size),
                                  initializer=tf.random_uniform_initializer(
                                      minval=-4 * np.sqrt(6. / self.config.hidden_size),
                                      maxval=4 * np.sqrt(6. / self.config.hidden_size)))
            W_x = tf.slice(W, [0, 0], [self.config.vocab_size, self.config.hidden_size])
            W_y = tf.slice(W, [self.config.vocab_size, 0], [self.config.vocab_size, self.config.hidden_size])
            # W_x = tf.get_variable('W_x', shape=(self.config.vocab_size, self.config.hidden_size),
            #                     initializer=tf.random_uniform(shape=(self.config.vocab_size, self.config.hidden_size),
            #                                                   minval=-4*np.sqrt(6. / self.config.hidden_size),
            #                                                   maxval=4*np.sqrt(6. / self.config.hidden_size)))
            # W_y = tf.get_variable('W_y', shape=(self.config.vocab_size, self.config.hidden_size),
            #                      initializer=tf.random_uniform(shape=(self.config.vocab_size, self.config.hidden_size),
            #                                                    minval=-4*np.sqrt(6. / self.config.hidden_size),
            #                                                    maxval=4*np.sqrt(6. / self.config.hidden_size)))
            b = tf.get_variable('b', shape=(self.config.hidden_size))
            b_x = tf.get_variable('b_x', shape=(self.config.hidden_size))
            b_y = tf.get_variable('b_y', shape=(self.config.hidden_size))

        # input_shape = tf.shape(x)
        with tf.variable_scope("decoder"):
            W_t = tf.transpose(W)
            W_xt = tf.transpose(W_x)
            W_yt = tf.transpose(W_y)
            b_t = tf.get_variable('b_t', shape=(2 * self.config.vocab_size))
            b_xt = tf.get_variable('b_xt', shape=(self.config.vocab_size))
            b_yt = tf.get_variable('b_yt', shape=(self.config.vocab_size))

        # a_x
        a_x = tf.nn.xw_plus_b(x, W_x, b_x)
        a_y = tf.nn.xw_plus_b(y, W_y, b_y)
        a_xy = tf.nn.xw_plus_b(xy, W, b)
        # h(a(x)) = hyperbolic of a(x)
        h_a_x = tf.nn.sigmoid(a_x)
        h_a_x_dropout = tf.nn.dropout(h_a_x, keep_prob=dropout_rate)
        h_a_y = tf.nn.sigmoid(a_y)
        h_a_y_dropout = tf.nn.dropout(h_a_y, keep_prob=dropout_rate)
        h_a_xy = tf.nn.sigmoid(a_xy)
        h_a_xy_dropout = tf.nn.dropout(h_a_xy, keep_prob=dropout_rate)

        # z1 = f(x, y), z2 = f(y, x), z3 = f(x), z4 = f(y), z5 = f((x,y), (x,y)), z6 = cor(a(x), a(y))
        z1 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_x_dropout, W_yt, b_yt))
        z2 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_y_dropout, W_xt, b_xt))
        z3 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_x_dropout, W_xt, b_xt))
        z4 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_y_dropout, W_yt, b_yt))
        # z5 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_y_dropout, W_yt, b_yt))
        z5 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_a_xy_dropout, W_t, b_t))

        xy_ = tf.concat([x, y], axis=0)
        mean_ = tf.reduce_mean(xy_, axis=1, keep_dims=True)
        cov_t = tf.matmul(xy_ - mean_, tf.transpose(xy_ - mean_)) / (self.config.vocab_size - 1)
        cov2_t = tf.diag(1 / tf.sqrt(tf.diag_part(cov_t)))
        cor = tf.matmul(cov2_t, tf.matmul(cov_t, cov2_t))
        z6 = tf.reduce_sum(cor)

        preds = [z1, z2, z3, z4, z5, z6]
        for pred in preds[:-2]:
            assert pred.get_shape().as_list() == [None, self.config.vocab_size], \
                "predictions are not of the right shape. Expected {}, got {}".format(
                    [None, self.config.vocab_size], pred.get_shape().as_list())
        self.tensor_dict['W'] = W

        # assert z5.get_shape().as_list() == [None, 2 * self.config.vocab_size], \
        #     "predictions are not of the right shape. Expected {}, got {}".format(
        #         [None, 2 * self.config.vocab_size], z5.get_shape().as_list())
        return preds

    def add_loss_op(self, preds):
        # preds = [(x, y), (y, x), (x), (y), ((x,y), (x, y)), cor(a(x), a(y))]
        x_labels = self.labels_placeholder1
        y_labels = self.labels_placeholder2
        xy_labels = tf.concat([x_labels, y_labels], axis=1)
        l1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[0], labels=y_labels), axis=1)
        #l1 = -tf.reduce_sum(y_labels * tf.log(tf.nn.sigmoid(preds[0])) + (1. - y_labels) * tf.log(1. - tf.nn.sigmoid(preds[0])), axis=1)
        l2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[1], labels=x_labels), axis=1)
        # l2 = -tf.reduce_sum(x_labels * tf.log(tf.nn.sigmoid(preds[1])) + (1. - x_labels) * tf.log(1. - tf.nn.sigmoid(preds[1])), axis=1)
        l3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[2], labels=x_labels), axis=1)
        # l3 = -tf.reduce_sum(x_labels * tf.log(tf.nn.sigmoid(preds[2])) + (1. - x_labels) * tf.log(1. - tf.nn.sigmoid(preds[2])), axis=1)
        l4 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[3], labels=y_labels), axis=1)
        # l4 = -tf.reduce_sum(y_labels * tf.log(tf.nn.sigmoid(preds[3])) + (1. - y_labels) * tf.log(1. - tf.nn.sigmoid(preds[3])), axis=1)
        l5 = tf.reduce_sum(self.config.beta * tf.nn.sigmoid_cross_entropy_with_logits(logits=preds[4], labels=xy_labels), axis=1)
        # l5 = -tf.reduce_sum(xy_labels * tf.log(tf.nn.sigmoid(preds[4])) + (1. - xy_labels) * tf.log(1. - tf.nn.sigmoid(preds[4])), axis=1)
        l6 = -tf.reduce_sum(self.config.lamda * preds[5])
        l = l1 + l2 + l3 + l4 + l5 + l6 + 100
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        l_ = [tf.reduce_mean(l1), tf.reduce_mean(l2), tf.reduce_mean(l3), tf.reduce_mean(l4), tf.reduce_mean(l5), tf.reduce_mean(l6)]
        return l, l_

    def add_cost_op(self, loss):
        cost = tf.reduce_mean(loss)
        return cost

    def add_training_op(self, cost):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(cost)
        return train_op

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss, self.loss_ = self.add_loss_op(self.pred)
        self.cost = self.add_cost_op(self.loss)
        self.train_op = self.add_training_op(self.cost)
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
        tf.summary.scalar('cost', self.cost)
        merged = tf.summary.merge_all()
        return merged

        # --- model operation ---

    # def output(self, sess, n_batch, inputs_batch, desc, data):
    #     preds = []
    #     pbar = tqdm(total=n_batch, desc=desc)
    #     for i, batch in enumerate(inputs_batch):
    #         # Ignore predict
    #         batch = batch[:1] + batch[2:]
    #         preds_ = self.predict_on_batch(sess, *batch)
    #         preds += list(preds_)
    #         pbar.update(1)

    # def evaluate(self, sess, data_loader):
    #     # --- train ---
    #     n_train_batch = data_loader.n_train_batch(self.config.batch_size)
    #     train_batch = data_loader.train_batch(self.config.batch_size)
    #     train_data = data_loader.train_data
    #     train_acc = self.output(sess, n_train_batch, train_batch, 'train eval', train_data)
    #
    #     # --- dev ---
    #     n_dev_batch = data_loader.n_dev_batch(self.config.batch_size)
    #     dev_batch = data_loader.dev_batch(self.config.batch_size)
    #     dev_data = data_loader.dev_data
    #     dev_acc = self.output(sess, n_dev_batch, dev_batch, 'dev eval', dev_data)
    #
    #     return train_acc, dev_acc

    # def predict_on_batch(self, sess, inputs_batch1, inputs_batch2):
    #     feed = self.create_feed_dict(inputs_batch1=inputs_batch1, inputs_batch2=inputs_batch2)
    #     predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
    #     return predictions

    def train_on_batch(self, sess, inputs_batch1, inputs_batch2, labels_batch1, labels_batch2):
        feed = self.create_feed_dict(inputs_batch1=inputs_batch1, inputs_batch2=inputs_batch2, labels_batch1=labels_batch1, labels_batch2=labels_batch2,
                                     dropout=self.config.dropout)
        _, cost, summary = sess.run([self.train_op, self.cost, self.sum_op], feed_dict=feed)
        #_, cost = sess.run([self.train_op, self.cost], feed_dict=feed)
        #return cost
        return cost, summary

    def run_epoch(self, sess, data, logger, writer, rand=True):
        x_data, y_data = data
        x_shp = x_data.shape
        y_shp = y_data.shape
        assert x_shp == y_shp

        logger.info('start epoch')
        # pbar = tqdm(total=x_shp[0])
        n_batch = int(math.ceil(x_shp[0] / self.config.batch_size))
        pbar = trange(n_batch)
        for i in pbar:
        # for i, batch in enumerate(data_loader.train_batch(self.config.batch_size)):
            if rand:
                rand_idx = np.random.permutation(x_shp[0])
                batch = (x_data[rand_idx[i: i+self.config.batch_size]], y_data[rand_idx[i: i+self.config.batch_size], :], x_data[rand_idx[i: i+self.config.batch_size]], y_data[rand_idx[i: i+self.config.batch_size]])
            else:
                batch = (x_data[i: i + self.config.batch_size], y_data[i: i + self.config.batch_size], x_data[i: i + self.config.batch_size], y_data[i: i + self.config.batch_size])
            #print(batch)
            cost, summary = self.train_on_batch(sess, *batch)
            #cost = self.train_on_batch(sess, *batch)
            writer.add_summary(summary, i)
            pbar.set_description("train loss = {}".format(cost))
            #pbar.update(1)
        print("")

    def fit(self, sess, saver, data, logger):
        writer = tf.summary.FileWriter(self.config.log_output + '/train',
                                             sess.graph)
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            self.run_epoch(sess, data, logger, writer)
            if saver:
                saver.save(sess, self.config.model_output)
            print("")

def run():
    # config = Config()
    pass


if __name__ == "__main__":
    # model = BilingualAutoencoder()
    print("hello")

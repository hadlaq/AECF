import numpy as np
import tensorflow as tf
import time

from data_manager import Data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def build_graph(X):
    d = 128
    n = 17770
    W1 = tf.get_variable(name="W1", shape=(d, n), initializer=tf.random_normal_initializer(0, 0.01))
    b1 = tf.get_variable(name="b1", shape=(d, 1), initializer=tf.zeros_initializer())
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)

    b2 = tf.get_variable(name="b2", shape=(n, 1), initializer=tf.zeros_initializer())
    Z2 = tf.matmul(W1, A1, transpose_a=True) + b2
    Y = tf.nn.sigmoid(Z2) * 5.0
    return Y


def get_next(iterator):
    X = iterator.get_next()
    X = tf.cast(X, tf.float32)
    X = tf.transpose(X)
    return X


def get_cost(X, Y):
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(X, zero)
    Xm = tf.cast(mask, tf.float32)
    Ym = tf.multiply(Y, Xm)
    ms = tf.reduce_sum(Xm, axis=0)
    loss = tf.divide(tf.reduce_sum(tf.square(Ym - X), axis=0), ms)
    return tf.reduce_mean(loss)


def train(dataobj, cost, epochs=100, lr=0.1):
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    train_summary = tf.summary.scalar("train loss", cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("../logs", graph=tf.get_default_graph())

        for e in range(epochs):
            train_epoch(e, sess, optimizer, train_summary, summary_writer, dataobj)


def train_epoch(epoch_num, sess, optimizer, train_summary, summary_writer, dataobj):
    dataobj.iterator_init(sess)
    print("Epoch: ", epoch_num + 1)
    chunk_num = 0
    done = False
    while not done:
        tic = time.time()
        if dataobj.is_done():
            done = True
        chunk_num += 1
        batch_num = 0
        total_cost = 0
        try:
            while True:
                _, current_cost, summary = sess.run([optimizer, cost, train_summary])
                total_cost += current_cost
                batch_num += 1
                # summary_writer.add_summary(summary, batch_num)
        except tf.errors.OutOfRangeError:
            dataobj.get_iterator()
            dataobj.iterator_init(sess)
        print("Epoch ", epoch_num + 1, " - chunk ", chunk_num, ":\t", total_cost / batch_num, "\t : ", time.time() - tic, "s")
    dataobj.new_epoch()
    dataobj.get_iterator()


epochs = 3
lr = 0.1

dataobj = Data(size=64)
iterator = dataobj.get_iterator()
X = get_next(iterator)
Y = build_graph(X)
cost = get_cost(X, Y)
train(dataobj, cost, epochs=epochs, lr=lr)
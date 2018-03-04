import tensorflow as tf
import numpy as np
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def activation(A, func):
    if func == 'selu':
        return tf.nn.selu(A)
    elif func == 'relu':
        return tf.nn.relu(A)
    elif func == 'sigmoid':
        return tf.nn.sigmoid(A) * 4.0 + 1
    else:
        return A


def initialize_weights(layers, constrained):
    W = {}
    b = {}
    init = tf.contrib.layers.xavier_initializer()
    for i in range(len(layers) - 1):
        W[i + 1] = tf.get_variable(name="W" + str(i + 1),
                                   shape=(layers[i + 1], layers[i]),
                                   initializer=init)
        if constrained:
            W[2 * len(layers) - 2 - i] = tf.transpose(W[i + 1])
        else:
            W[2 * len(layers) - 2 - i] = tf.get_variable(name="W" + str(2 * len(layers) - 2 - i),
                                                         shape=(layers[i], layers[i + 1]),
                                                         initializer=init)
        b[i + 1] = tf.get_variable(name="b" + str(i + 1),
                                   shape=(layers[i + 1], 1),
                                   initializer=tf.zeros_initializer())
        b[2 * len(layers) - 2 - i] = tf.get_variable(name="b" + str(2 * len(layers) - 2 - i),
                                                     shape=(layers[i], 1),
                                                     initializer=tf.zeros_initializer())
    return W, b


def encoder(A, weights, func, keep_prob=1.0):
    W, b = weights
    layers = int(len(W) / 2)
    for layer in range(1, layers + 1):
        A = activation(tf.matmul(W[layer], A) + b[layer], func)
        if 0.999 > keep_prob:
            A = tf.nn.dropout(A, keep_prob)
    return A


def decoder(A, weights, func, last_func):
    W, b = weights
    layers = int(len(W) / 2)
    for layer in range(layers + 1, 2 * layers):
        A = activation(tf.matmul(W[layer], A) + b[layer], func)
    A = activation(tf.matmul(W[2 * layers], A) + b[2 * layers], last_func)
    return A


def autoencoder(X, layers, constrained=True, keep_prob=1.0, func="selu", last_func="selu"):
    weights = initialize_weights(layers, constrained)
    A = encoder(X, weights, func, keep_prob)
    A = decoder(A, weights, func, last_func)
    return A


def get_loss(Y, Yhat):
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(Y, zero)
    mask = tf.cast(mask, tf.float32)
    Yhatm = tf.multiply(Yhat, mask)
    loss = tf.reduce_sum(tf.square(Yhatm - Y))
    mask = tf.reduce_sum(mask)
    loss = tf.divide(loss, mask)
    return loss


def get_optimizer(optimizer_type, lr, momentum):
    if optimizer_type == "momentum":
        return tf.train.MomentumOptimizer(lr, momentum)
    elif optimizer_type == "adam":
        return tf.train.AdamOptimizer(lr)
    else:
        return tf.train.GradientDescentOptimizer(lr)


def train(data_train, data_dev, loss, optimizer, X, Y, Yhat, epochs=50, lr=0.1, momentum=0.9, dense_refeeding=False):
    train_losses = []
    eval_losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            train_loss = train_epoch(e, sess, optimizer, loss, data_train, X, Y, Yhat, train_losses, dense_refeeding)
            train_losses.append(train_loss)

            eval_loss = eval_epoch(e, sess, loss, data_dev, X, Y, eval_costs)
            eval_losses.append(eval_loss)

    return train_losses, eval_losses


def train_epoch(epoch_num, sess, optimizer, loss, data_train, X, Y, Yhat, dense_refeeding):
    batches = 0
    total_loss = 0
    tic = time.time()
    while True:
        x = data_train.get_next()
        if x is None:
            # finished epoch
            data_train.new_epoch()
            break

        batches += 1
        _, current_loss, fx = sess.run([optimizer, loss, Yhat], feed_dict={X: x, Y: x})
        total_loss += current_loss

        if dense_refeeding:
            batches += 1
            _, current_loss, summary = sess.run([optimizer, loss], feed_dict={X: fx, Y: fx})
            total_loss += current_loss

    epoch_loss = np.sqrt(total_loss / batches)
    print("Train epoch " + str(epoch_num + 1) + ":\t", epoch_loss, "\t : ", time.time() - tic, "s")
    return epoch_loss


def eval_epoch(epoch_num, sess, cost, data_dev, X, Y):
    batches = 0
    total_loss = 0
    tic = time.time()
    while True:
        point = data_dev.get_next()
        if point is None:
            # finished epoch
            data_dev.new_epoch()
            break

        x, y = point
        batches += 1
        current_loss = sess.run([cost], feed_dict={X: x, Y: y})
        total_loss += current_loss

    epoch_loss = np.sqrt(total_loss / batches)
    print("Eval epoch " + str(epoch_num + 1) + ":\t", epoch_loss, "\t : ", time.time() - tic, "s")
    return epoch_loss

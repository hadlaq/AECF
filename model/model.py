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


def autoencoder(X, layers, constrained=True, keep_prob=1.0, func="selu", last_func="selu", weights=None):
    if weights is None:
        weights = initialize_weights(layers, constrained)
    A = encoder(X, weights, func, keep_prob)
    A = decoder(A, weights, func, last_func)
    return A, weights


def get_loss(Y, Yhat):
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(Y, zero)
    mask = tf.cast(mask, tf.float32)
    Yhatm = tf.multiply(Yhat, mask)
    loss = tf.reduce_sum(tf.square(Yhatm - Y))
    mask = tf.reduce_sum(mask)
    loss = tf.divide(loss, mask)
    return loss


def get_test_loss(Y, Yhat):
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(Y, zero)
    mask = tf.cast(mask, tf.float32)
    Yhatm = tf.multiply(Yhat, mask)
    loss = tf.reduce_sum(tf.square(Yhatm - Y))
    mask = tf.reduce_sum(mask)
    return loss, mask


def get_optimizer(optimizer_type, lr, momentum):
    if optimizer_type == "momentum":
        return tf.train.MomentumOptimizer(lr, momentum)
    elif optimizer_type == "adam":
        return tf.train.AdamOptimizer(lr)
    else:
        return tf.train.GradientDescentOptimizer(lr)


def train(data_train, data_dev, losses, optimizer, X, Y, Yhat, epochs=50,
          dense_refeeding=False):
    loss, loss_sum, loss_examples, loss_sum_dev, loss_examples_dev = losses
    train_step = optimizer.minimize(loss)
    saver = tf.train.Saver()

    best_eval = 99999999.0
    train_losses = []
    eval_losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            train_loss = train_epoch(e, sess, train_step, loss_sum, loss_examples, data_train, X, Y, Yhat, dense_refeeding)
            train_losses.append(train_loss)

            eval_loss = eval_epoch(e, sess, loss_sum_dev, loss_examples_dev, data_dev, X, Y)
            eval_losses.append(eval_loss)
            if eval_loss < best_eval:
                best_eval = eval_loss
                saver.save(sess, 'checkpoint/model')



    return train_losses, eval_losses


def train_epoch(epoch_num, sess, train_step, loss_sum, loss_examples, data_train, X, Y, Yhat, dense_refeeding):
    total_loss = 0
    total_examples = 0
    tic = time.time()
    while True:
        x = data_train.get_next()
        if x is None:
            # finished epoch
            data_train.new_epoch()
            break

        _, current_loss, current_examples, fx = sess.run([train_step, loss_sum, loss_examples, Yhat], feed_dict={X: x, Y: x})
        total_loss += current_loss
        total_examples += current_examples

        if dense_refeeding:
            _, current_loss, current_examples = sess.run([train_step, loss_sum, loss_examples], feed_dict={X: fx, Y: fx})
            total_loss += current_loss
            total_examples += current_examples

    epoch_loss = np.sqrt(total_loss / total_examples)
    print("Train epoch " + str(epoch_num + 1) + ":\t", epoch_loss, "\t : ", time.time() - tic, "s")
    return epoch_loss


def eval_epoch(epoch_num, sess, loss_sum_dev, loss_examples_dev, data_dev, X, Y):
    total_loss = 0
    total_examples = 0
    tic = time.time()
    while True:
        point = data_dev.get_next()
        if point is None:
            # finished epoch
            data_dev.new_epoch()
            break

        x, y = point
        current_loss, current_examples = sess.run([loss_sum_dev, loss_examples_dev], feed_dict={X: x, Y: y})
        total_loss += current_loss
        total_examples += current_examples

    epoch_loss = np.sqrt(total_loss / total_examples)
    print("Eval epoch " + str(epoch_num + 1) + ":\t", epoch_loss, "\t : ", time.time() - tic, "s")
    return epoch_loss


def test(data, X, Y, YhatDev):
    total_loss = 0
    total_examples = 0
    tic = time.time()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("loaded model")
        while True:
            point = data.get_next()
            if point is None:
                # finished epoch
                break
            x, y = point
            fx, = sess.run([YhatDev], feed_dict={X: x, Y: y})
            current_loss, current_examples = loss(y, fx)
            total_loss += current_loss
            total_examples += current_examples

        RMSE = np.sqrt(total_loss / total_examples)
        print("Test ", RMSE, "\t : ", time.time() - tic, "s")

def loss(r, y):
    m = (r != 0).astype(float)
    losses = np.sum(m * np.square(r - y))
    ratings = np.sum(m)
    return losses, ratings
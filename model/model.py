import tensorflow as tf
import time
import matplotlib.pyplot as plt

from data_manager import Data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def last_activation(A):
    return activation(A)


def activation(A):
    return tf.nn.relu(A)


def initialize_weights(layers, constrained):
    W = {}
    b = {}
    for i in range(len(layers) - 1):
        W[i + 1] = tf.get_variable(name="W" + str(i + 1),
                               shape=(layers[i + 1], layers[i]),
                               initializer=tf.random_normal_initializer(0, 0.01))
        if constrained:
            W[2 * len(layers) - 2 - i] = tf.transpose(W[i + 1])
        else:
            W[2 * len(layers) - 2 - i] = tf.get_variable(name="W" + str(2 * len(layers) - 2 - i),
                                                         shape=(layers[i], layers[i + 1]),
                                                         initializer=tf.random_normal_initializer(0, 0.01))
        b[i + 1] = tf.get_variable(name="b" + str(i + 1),
                               shape=(layers[i + 1], 1),
                               initializer=tf.zeros_initializer())
        b[2 * len(layers) - 2 - i] = tf.get_variable(name="b" + str(2 * len(layers) - 2 - i),
                                                     shape=(layers[i], 1),
                                                     initializer=tf.zeros_initializer())
    return W, b


def encoder(A, weights):
    W, b = weights
    layers = int(len(W) / 2)
    for layer in range(1, layers + 1):
        A = activation(tf.matmul(W[layer], A) + b[layer])
    return A


def decoder(A, weights):
    W, b = weights
    layers = int(len(W) / 2)
    for layer in range(layers + 1, 2 * layers):
        A = activation(tf.matmul(W[layer], A) + b[layer])
    A = activation(tf.matmul(W[2 * layers], A) + b[2 * layers])
    return A


def autoencoder(X, layers, constrained=True):
    weights = initialize_weights(layers, constrained)
    A = encoder(X, weights)
    A = decoder(A, weights)
    return A


def get_cost(X, Y):
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(X, zero)
    Xm = tf.cast(mask, tf.float32)
    Ym = tf.multiply(Y, Xm)
    ms = tf.reduce_sum(Xm, axis=0)
    loss = tf.divide(tf.reduce_sum(tf.square(Ym - X), axis=0), ms)
    return tf.reduce_mean(loss)


def train(dataobj, dataobjdev, cost, X, Y, epochs=100, lr=0.1):
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    train_summary = tf.summary.scalar("train loss", cost)
    eval_summary = tf.summary.scalar("eval loss", cost)
    train_costs = []
    eval_costs = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("../logs", graph=tf.get_default_graph())

        train_step = 0
        eval_step = 0
        for e in range(epochs):
            train_step = train_epoch(e, train_step, sess, optimizer, cost, train_summary, summary_writer, dataobj, X, Y, train_costs)
            eval_step = eval_epoch(e, eval_step, sess, cost, eval_summary, summary_writer, dataobjdev, X, Y, eval_costs)
    return train_costs, eval_costs


def train_epoch(epoch_num, step, sess, optimizer, cost, train_summary, summary_writer, dataobj, X, Y, train_costs):
    batch = 0
    total_cost = 0
    tic = time.time()
    while True:
        x = dataobj.get_next()
        if x is None:
            break
        batch += 1
        step += 1
        _, current_cost, summary = sess.run([optimizer, cost, train_summary], feed_dict={X: x, Y: x})
        total_cost += current_cost
        summary_writer.add_summary(summary, step)
    print("Train epoch ", epoch_num + 1, ":\t", total_cost / batch, "\t : ", time.time() - tic, "s")
    train_costs.append(total_cost / batch)
    dataobj.new_epoch()
    return step


def eval_epoch(epoch_num, step, sess, cost, eval_summary, summary_writer, dataobjdev, X, Y, eval_costs):
    batch = 0
    total_cost = 0
    tic = time.time()
    while True:
        point = dataobjdev.get_next()
        if point is None:
            break
        x, y = point
        batch += 1
        step += 1
        current_cost, summary = sess.run([cost, eval_summary], feed_dict={X: x, Y: y})
        total_cost += current_cost
        summary_writer.add_summary(summary, step)
    print("Eval epoch ", epoch_num + 1, ":\t", total_cost / batch, "\t : ", time.time() - tic, "s")
    eval_costs.append(total_cost / batch)
    dataobjdev.new_epoch()
    return step

epochs = 50
lr = 1
batch_size = 32

dataobj = Data(size=512, batch=batch_size)
dataobjdev = Data(size=512, batch=batch_size, path="../data/netflix/output_small_dev", test=True)
X = tf.placeholder(tf.float32, [17770, None], name='X')
Y = tf.placeholder(tf.float32, [17770, None], name='Y')
layers = [17770, 256, 128]
Yhat = autoencoder(X, layers)
cost = get_cost(Y, Yhat)
train_costs, eval_costs = train(dataobj, dataobjdev, cost, X, Y, epochs=epochs, lr=lr)

t, = plt.plot([i+1 for i in range(epochs)], train_costs, label="Train")
e, = plt.plot([i+1 for i in range(epochs)], eval_costs, label="Dev")
plt.legend(handles=[t, e])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()

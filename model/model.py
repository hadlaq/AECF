import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Will change this later but this is an example of how data can be consumed.
def parse_data_point(line):
    n = 17770
    line = line.strip()
    user, temp = line.split(":")
    ratings = temp.split(" ")
    ratings = [tuple(r.split(",")) for r in ratings]
    vector = np.zeros(n)
    for r in ratings:
        movie, rating, date = r
        vector[int(movie) - 1] = float(rating)
    return vector


def get_data(m, tm):
    i = 0
    n = 17770
    matrix = np.zeros((n, m+tm))
    with open("../data/netflix/output") as f:
        for line in f:
            matrix[:, i] = parse_data_point(line)
            i += 1
            if m+tm == i:
                break
    train_mat = matrix[:, 0:m]
    validation = matrix[:, m:m+tm]

    training_dataset = tf.data.Dataset.from_tensor_slices(train_mat.T)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation.T)

    training_dataset.batch(32)
    validation_dataset.batch(32)

    it = tf.data.Iterator.from_structure(training_dataset.output_types,
                                         training_dataset.output_shapes)

    training_init_op = it.make_initializer(training_dataset)
    validation_init_op = it.make_initializer(validation_dataset)

    return it, training_init_op, validation_init_op

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
    X = tf.reshape(X, [-1, 1])
    return X


def get_cost(X, Y):
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(X, zero)
    Xm = tf.cast(mask, tf.float32)
    Ym = tf.multiply(Y, Xm)
    ms = tf.reduce_sum(Xm, axis=0)
    loss = tf.divide(tf.reduce_sum(tf.square(Ym - X), axis=0), ms)
    return tf.reduce_mean(loss), Ym


def train(cost, Ym, epochs=100, lr=0.1):
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    train_summary = tf.summary.scalar("train loss", cost)
    validation_summary = tf.summary.scalar("validation loss", cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("../logs", graph=tf.get_default_graph())

        i = 0
        for e in range(epochs):
            sess.run(training_init_op)
            total = 0
            try:
                while True:
                    _, current_cost, summary = sess.run([optimizer, cost, train_summary])
                    total += current_cost
                    summary_writer.add_summary(summary, i)
                    i += 1
            except tf.errors.OutOfRangeError:
                pass
            print("Train Epoch ", e + 1, ":\t", total / 6000.0)

            sess.run(validation_init_op)
            total = 0

            try:
                while True:
                    current_cost, summary = sess.run([cost, validation_summary])
                    total += current_cost
                    i += 1
            except tf.errors.OutOfRangeError:
                pass
            print("Valid Epoch ", e + 1, ":\t", total / 100.0)


training_size = 6000
validation_size = 100

epochs = 100
lr = 0.1

iterator, training_init_op, validation_init_op = get_data(training_size, validation_size)
X = get_next(iterator)
Y = build_graph(X)
cost = get_cost(X, Y)
train(cost, epochs=epochs, lr=lr)

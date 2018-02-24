import numpy as np
import tensorflow as tf


def parse_ratings(temp):
    n = 17770
    ratings = temp.split(" ")
    ratings = [tuple(r.split(",")) for r in ratings]
    vector = np.zeros(n)
    for r in ratings:
        movie, rating, date = r
        vector[int(movie) - 1] = float(rating)
    return vector


def parse_data_point(line):
    line = line.strip()
    user, temp = line.split(":")
    return parse_ratings(temp)


def parse_test_point(line):
    line = line.strip()
    user, tempi, tempt = line.split(":")
    return parse_ratings(tempi), parse_ratings(tempt)


class Data:
    def __init__(self, size=1024, batch=32, path="../data/netflix/output_small_train", test=False):
        self.offset = 0
        self.size = size
        self.batch = batch
        self.path = path
        self.read_points = 0
        self.done = False
        self.iterator = None
        self.initializer = None
        self.test = test

    def get_iterator(self):
        print("getting data")
        n = 17770
        matrix = np.zeros((n, self.size))
        if self.test:
            matrix_test = np.zeros((n, self.size))
        f = open(self.path)
        j = 0
        for i in range(self.size):
            j = i
            self.read_points += 1
            f.seek(self.offset)
            line = f.readline()
            self.offset = f.tell()
            if len(line) == 0:
                self.done = True
                break
            if self.test:
                inp, target = parse_test_point(line)
                matrix[:, i] = inp
                matrix_test[:, i] = target
            else:
                matrix[:, i] = parse_data_point(line)
        matrix = matrix[:, 0:j]
        if self.test:
            matrix_test = matrix_test[:, 0:j]
        f.close()

        print("got data")
        if self.test:
            dataset = tf.data.Dataset.from_tensor_slices((matrix.T, matrix_test.T))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(matrix.T)

        dataset = dataset.batch(self.batch)

        if self.iterator is None:
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        else:
            iterator = self.iterator

        init = iterator.make_initializer(dataset)

        self.iterator = iterator
        self.initializer = init

        print("really got data")
        return iterator

    def iterator_init(self, sess):
        print("1 crashed?")
        sess.run(self.initializer)
        print("2 crashed?")

    def is_done(self):
        return self.done

    def new_epoch(self):
        self.offset = 0
        self.read_points = 0
        self.done = False
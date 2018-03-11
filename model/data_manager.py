import numpy as np


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
    user, tempt, tempi = line.split(":")
    return parse_ratings(tempi), parse_ratings(tempt)


class Data:
    def __init__(self, size=1024, batch=32, path="../data/netflix/output_small_train", test=False):
        self.offset = 0
        self.size = size
        self.batch = batch
        self.path = path
        self.done = False
        self.matrix = None
        self.read = 0
        self.test = test

    def get_next(self):
        if self.done:
            return None
        if self.matrix is None or self.read >= self.matrix.shape[1]:
            self._load_matrix()
            self.read = 0
        if self.read + self.batch <= self.matrix.shape[1]:
            current = self.read
            self.read += self.batch
            if self.test:
                return self.matrix[:, current:self.read, 0], self.matrix[:, current:self.read, 1]
            else:
                return self.matrix[:, current:self.read]
        else:
            current = self.read
            self.read = self.matrix.shape[1]
            if self.test:
                return self.matrix[:, current:, 0], self.matrix[:, current:, 1]
            else:
                return self.matrix[:, current:]

    def _load_matrix(self):
        n = 17770
        if self.test:
            matrix = np.zeros((n, self.size, 2))
        else:
            matrix = np.zeros((n, self.size))
        f = open(self.path)
        j = 0
        for i in range(self.size):
            f.seek(self.offset)
            line = f.readline()
            self.offset = f.tell()
            if len(line) == 0:
                self.done = True
                break
            j = i
            if self.test:
                inp, target = parse_test_point(line)
                matrix[:, i, 0] = inp
                matrix[:, i, 1] = target
            else:
                matrix[:, i] = parse_data_point(line)
        if self.test:
            self.matrix = matrix[:, 0:j + 1, :]
        else:
            self.matrix = matrix[:, 0:j + 1]

    def is_done(self):
        return self.done

    def new_epoch(self):
        self.offset = 0
        self.read = 0
        self.done = False
        self.matrix = None
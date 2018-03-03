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


def to_tensor(line):
    n = 17770
    line = line.strip()
    user, temp = line.split(":")
    ratings = temp.split(" ")
    ratings = [tuple(r.split(",")) for r in ratings]
    vector = np.zeros(n)
    for r in ratings:
        movie, rating, date = r
        vector[int(movie) - 1] = float(rating)
    return tf.convert_to_tensor(vector, tf.float32)


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def write_tfr():
    i = 0
    writer = tf.python_io.TFRecordWriter("../data/netflix/tfr_small")
    with open("../data/netflix/output") as f:
        for line in f:
            i += 1
            v = parse_data_point(line)
            t = floats_feature(v)
            writer.write(t.SerializeToString())
            if i == 10000:
                break
    writer.close()

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

    # training_dataset = tf.data.Dataset.from_tensor_slices(train_mat.T)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation.T)

    training_dataset = tf.data.TextLineDataset("../data/netflix/output")
    training_dataset = training_dataset.map(
    lambda line: tuple(tf.py_func(to_tensor, [line], [str])))
    training_dataset = training_dataset.map(to_tensor)

    training_dataset = training_dataset.batch(32)
    validation_dataset = validation_dataset.batch(32)
    it = tf.data.Iterator.from_structure(training_dataset.output_types,
                                         training_dataset.output_shapes)

    training_init_op = it.make_initializer(training_dataset)
    validation_init_op = it.make_initializer(validation_dataset)

    return it, training_init_op, validation_init_op

def get_data2():
    # training_dataset = tf.data.TFRecordDataset("../data/netflix/tfr_small")
    training_dataset = tf.data.TextLineDataset("../data/netflix/output")
    training_dataset = training_dataset.map(
        lambda line: tuple(tf.py_func(to_tensor, [line], [tf.string]))
    )

    training_dataset = training_dataset.batch(32)
    it = tf.data.Iterator.from_structure(training_dataset.output_types,
                                         training_dataset.output_shapes)
    training_init_op = it.make_initializer(training_dataset)
    return it, training_init_op
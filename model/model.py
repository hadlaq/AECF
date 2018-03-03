import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from data_manager import Data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def last_activation(A):
    return activation(A)
    # return tf.nn.sigmoid(A) * 4.0 + 1


def activation(A):
    return tf.nn.relu(A)


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


def encoder(A, weights, keep_prob=1.0):
    W, b = weights
    layers = int(len(W) / 2)
    for layer in range(1, layers + 1):
        A = activation(tf.matmul(W[layer], A) + b[layer])
        if keep_prob < 0.999:
            A = tf.nn.dropout(A, keep_prob)
    return A


def decoder(A, weights):
    W, b = weights
    layers = int(len(W) / 2)
    for layer in range(layers + 1, 2 * layers):
        A = activation(tf.matmul(W[layer], A) + b[layer])
    A = last_activation(tf.matmul(W[2 * layers], A) + b[2 * layers])
    return A


def autoencoder(X, layers, constrained=True, keep_prob=1.0):
    weights = initialize_weights(layers, constrained)
    A = encoder(X, weights, keep_prob)
    A = decoder(A, weights)
    return A


def get_cost(X, Y):
    zero = tf.constant(0, dtype=tf.float32)
    mask = tf.not_equal(X, zero)
    Xm = tf.cast(mask, tf.float32)
    Ym = tf.multiply(Y, Xm)
    ms = tf.reduce_sum(Xm, axis=0)
    loss = tf.reduce_sum(tf.square(Ym - X), axis=0)
    # loss = tf.divide(tf.reduce_sum(tf.square(Ym - X), axis=0), ms)
    return tf.reduce_mean(loss)


def get_optimizer(lr, momentum):
    return tf.train.MomentumOptimizer(lr, momentum)
    # return tf.train.GradientDescentOptimizer(lr)


def train(dataobj, dataobjdev, cost, X, Y, Yhat, epochs=100, lr=0.1, momentum=0.9, dense_refeeding=True):
    optimizer = get_optimizer(lr, momentum).minimize(cost)
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
            train_step = train_epoch(e, train_step, sess, optimizer, cost, train_summary, summary_writer, dataobj, X, Y, Yhat, train_costs, dense_refeeding)
            eval_step = eval_epoch(e, eval_step, sess, cost, eval_summary, summary_writer, dataobjdev, X, Y, eval_costs)
    return train_costs, eval_costs


def train_epoch(epoch_num, step, sess, optimizer, cost, train_summary, summary_writer, dataobj, X, Y, Yhat, train_costs, dense_refeeding):
    batch = 0
    total_cost = 0
    tic = time.time()
    zeros = 0
    while True:
        x = dataobj.get_next()
        if x is None:
            break
        batch += 1
        step += 1
        fx, _, current_cost, summary = sess.run([Yhat, optimizer, cost, train_summary], feed_dict={X: x, Y: x})
        total_cost += current_cost
        summary_writer.add_summary(summary, step)
        # print(current_cost)
        if dense_refeeding:
            s = np.sum(fx)
            if s < 0.0001:
                zeros += 1
            else:
                batch += 1
                step += 1
                _, current_cost, summary = sess.run([optimizer, cost, train_summary], feed_dict={X: fx, Y: fx})
                total_cost += current_cost
                summary_writer.add_summary(summary, step)
            # print(current_cost)
    print(zeros, "/", batch)
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

epochs = 100
# lr = 1
lr = 0.005
batch_size = 32
# batch_size = 128
dropout = 0.8
# dropout = 0.0
keep_prob = 1.0 - dropout

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataobj = Data(size=512, batch=batch_size, path="../data/netflix/output_small_train")
dataobjdev = Data(size=512, batch=batch_size, path="../data/netflix/output_small_dev", test=True)
X = tf.placeholder(tf.float32, [17770, None], name='X')
Y = tf.placeholder(tf.float32, [17770, None], name='Y')
layers = [17770, 512, 512, 1024]
# layers = [17770, 128]
Yhat = autoencoder(X, layers, keep_prob=keep_prob)
cost = get_cost(Y, Yhat)
train_costs, eval_costs = train(dataobj, dataobjdev, cost, X, Y, Yhat, epochs=epochs, lr=lr, dense_refeeding=True)

t, = plt.plot([i+1 for i in range(epochs)], train_costs, label="Train")
e, = plt.plot([i+1 for i in range(epochs)], eval_costs, label="Dev")
plt.legend(handles=[t, e])
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()

# graph build
# Train epoch  1 :	 1.05368708977 	 :  50.23587489128113 s
# Eval epoch  1 :	 1.42984764458 	 :  11.19926404953003 s
# Train epoch  2 :	 0.895219954305 	 :  49.405231952667236 s
# Eval epoch  2 :	 1.51232831841 	 :  11.080128908157349 s

# transpose_a
# Train epoch  1 :	 8.78181792871 	 :  49.08986735343933 s
# Eval epoch  1 :	 6.36020069681 	 :  10.954441785812378 s
# Train epoch  2 :	 3.88312153439 	 :  49.611257791519165 s
# Eval epoch  2 :	 4.86192533385 	 :  10.903506517410278 s

# no dropout tag
# Train epoch  1 :	 8.448819705 	 :  49.08167123794556 s
# Eval epoch  1 :	 6.25568258195 	 :  10.962835311889648 s
# Train epoch  2 :	 3.72955420274 	 :  49.04672646522522 s
# Eval epoch  2 :	 4.87514740381 	 :  11.143251657485962 s

# last activation
# Train epoch  1 :	 1.0517946441 	 :  50.679285526275635 s
# Eval epoch  1 :	 1.44149944983 	 :  11.274213552474976 s
# Train epoch  2 :	 0.894281404178 	 :  53.77819848060608 s
# Eval epoch  2 :	 1.48905721143 	 :  11.262027740478516 s

# tf.transpose
# Train epoch  1 :	 1.05174345707 	 :  49.43943643569946 s
# Eval epoch  1 :	 1.44263179071 	 :  10.93159818649292 s
# Train epoch  2 :	 0.893917043901 	 :  50.11503553390503 s
# Eval epoch  2 :	 1.50330438981 	 :  11.233317613601685 s

# dropout tag only
# Train epoch  1 :	 1.05189970875 	 :  49.79075765609741 s
# Eval epoch  1 :	 1.4418184159 	 :  10.924262762069702 s

# non constrained!!!!!!!!!!!!!
# Train epoch  1 :	 0.992327292965 	 :  48.64381241798401 s
# Eval epoch  1 :	 1.10124486772 	 :  10.891397953033447 s
# Train epoch  2 :	 0.818839704 	 :  49.650235652923584 s
# Eval epoch  2 :	 1.20567865468 	 :  11.395399808883667 s
# Train epoch  3 :	 0.730226142324 	 :  51.330995082855225 s
# Eval epoch  3 :	 1.25974195737 	 :  11.932974338531494 s

# momentum, lr = 0.005
# Train epoch  1 :	 1.19466777874 	 :  50.82901954650879 s
# Eval epoch  1 :	 1.1905518934 	 :  11.312828779220581 s
# Train epoch  2 :	 1.06721916052 	 :  50.684372663497925 s
# Eval epoch  2 :	 1.17021427093 	 :  11.010853052139282 s

# layers all
# Train epoch  1 :	 1.0867602667 	 :  58.702738761901855 s
# Eval epoch  1 :	 1.06074831621 	 :  14.251103162765503 s
# Train epoch  2 :	 0.893754497813 	 :  58.33870887756348 s
# Eval epoch  2 :	 1.08937391257 	 :  11.41596007347107 s
# Train epoch  3 :	 0.813025225498 	 :  63.13024115562439 s
# Eval epoch  3 :	 1.15286038276 	 :  12.220107555389404 s
# Train epoch  4 :	 0.751387106436 	 :  112.95071816444397 s
# Eval epoch  4 :	 1.22875099392 	 :  17.257349729537964 s

# activation relu all
# Train epoch  1 :	 4.67690526756 	 :  71.45280408859253 s
# Eval epoch  1 :	 4.86920159902 	 :  12.751112222671509 s
# Train epoch  2 :	 3.89931377793 	 :  59.279810190200806 s
# Eval epoch  2 :	 4.81405665936 	 :  11.989749908447266 s
# Train epoch  3 :	 3.79498722886 	 :  58.95450305938721 s
# Eval epoch  3 :	 4.86260571497 	 :  11.227056741714478 s
# Train epoch  4 :	 3.73016863029 	 :  59.668832302093506 s
# Eval epoch  4 :	 4.89792441448 	 :  12.094951152801514 s
# Train epoch  5 :	 3.69824268491 	 :  59.21188926696777 s
# Eval epoch  5 :	 4.8359102312 	 :  12.13498330116272 s
# Train epoch  6 :	 3.66052053993 	 :  62.1399564743042 s
# Eval epoch  6 :	 4.90055639491 	 :  12.389582872390747 s
# Train epoch  7 :	 3.6282797242 	 :  59.79430317878723 s
# Eval epoch  7 :	 5.00261731899 	 :  11.949129819869995 s
# Train epoch  8 :	 3.59622699663 	 :  60.44804406166077 s
# Eval epoch  8 :	 4.89723657077 	 :  12.853363513946533 s
# Train epoch  9 :	 3.58162341406 	 :  60.55968451499939 s
# Eval epoch  9 :	 4.88331738758 	 :  12.061825513839722 s
# Train epoch  10 :	 3.55744634132 	 :  58.94528555870056 s
# Eval epoch  10 :	 5.13991633352 	 :  12.077388048171997 s
# Train epoch  11 :	 3.54277343222 	 :  58.83842658996582 s
# Eval epoch  11 :	 5.35337307165 	 :  12.01993703842163 s
# Train epoch  12 :	 3.5339369691 	 :  58.85137152671814 s
# Eval epoch  12 :	 5.29705618939 	 :  12.050177812576294 s
# Train epoch  13 :	 3.51958098031 	 :  58.910022497177124 s
# Eval epoch  13 :	 5.01053103685 	 :  11.42306399345398 s

# Dense refeeding
# Train epoch  1 :	 nan 	 :  80.16165351867676 s
# Eval epoch  1 :	 14.3054864101 	 :  11.203904390335083 s
# Train epoch  2 :	 nan 	 :  78.58519983291626 s
# Eval epoch  2 :	 14.3054864101 	 :  11.241914510726929 s
# Train epoch  3 :	 nan 	 :  79.45408272743225 s
# Eval epoch  3 :	 14.3054864101 	 :  12.062618970870972 s
# Train epoch  4 :	 nan 	 :  81.3620195388794 s
# Eval epoch  4 :	 14.3054864101 	 :  12.123862266540527 s
# Train epoch  5 :	 nan 	 :  79.38319563865662 s
# Eval epoch  5 :	 14.3054864101 	 :  11.181966781616211 s

# Cancel

# Batch size 128
# InternalError (see above for traceback): Blas GEMM launch failed : a.shape=(512, 17770), b.shape=(17770, 128), m=512, n=128, k=17770
# 	 [[Node: MatMul = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](W1/read, _arg_X_0_0/_1)]]
# 	 [[Node: Relu_5/_19 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_460_Relu_5", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

# Batch size 64
# Train epoch  1 :	 nan 	 :  66.77734327316284 s
# Eval epoch  1 :	 14.3141300591 	 :  11.60570216178894 s

# cancel
# Train epoch  1 :	 3.82580530713 	 :  59.1991765499115 s
# Eval epoch  1 :	 4.22990233383 	 :  12.396090984344482 s
# Train epoch  2 :	 3.24475095476 	 :  59.49044322967529 s
# Eval epoch  2 :	 4.23943087994 	 :  11.429962158203125 s

# dropout 0.8
# Train epoch  1 :	 8.74070963549 	 :  59.690995931625366 s
# Eval epoch  1 :	 7.38946878517 	 :  11.201585292816162 s
# Train epoch  2 :	 1.31112975667e+30 	 :  57.31390857696533 s
# Eval epoch  2 :	 14.3054864101 	 :  11.236913204193115 s
# Train epoch  3 :	 14.7445128472 	 :  58.67403769493103 s
# Eval epoch  3 :	 14.3054864101 	 :  12.143005609512329 s
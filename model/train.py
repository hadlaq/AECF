import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from model import model
from model import data_manager


def parse_args():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability=1-keep_prob')
    parser.add_argument('--layers', nargs='+', type=int, default=[128, 256], help='description of the encoder layers')
    parser.add_argument('--constrained', type=bool, default=True, help='constrained autoencoder')
    parser.add_argument('--activation', type=str, default="selu", help='activation function (selu, relu, sigmoid)')
    parser.add_argument('--last_activation', type=str, default="selu",
                        help='activation function of last layer (selu, relu, sigmoid)')
    parser.add_argument('--optimizer', type=str, default="momentum", help='momentum, adam, gd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum parameter')
    parser.add_argument('--dense_refeeding', type=bool, default=True, help='does dense refeeding')

    # training params
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--chunk_size', type=int, default=512, help='number of data points to read from disk each time')
    parser.add_argument('--small_dataset', type=bool, default=True, help='use the small dataset')

    return parser.parse_args()


def main():
    args = parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    X = tf.placeholder(tf.float32, [17770, None], name='X')
    Y = tf.placeholder(tf.float32, [17770, None], name='Y')
    Yhat = model.autoencoder(X, args.layers, keep_prob=(1.0 - args.dropout), constrained=args.constrained)
    loss = model.get_loss(Y, Yhat)
    optimizer = model.get_optimizer(args.optimizer_type, args.lr, args.momentum)

    if args.small_dataset:
        train_path = "../data/netflix/output_small_train"
        dev_path = "../data/netflix/output_small_dev"
    else:
        train_path = "../data/netflix/output_train"
        dev_path = "../data/netflix/output_dev"

    data_train = data_manager.Data(size=args.chunk_size, batch=args.batch_size, path=train_path)
    data_dev = data_manager.Data(size=args.chunk_size, batch=args.batch_size, path=dev_path, test=True)

    train_losses, eval_losses = model.train(data_train, data_dev, loss, optimizer, X, Y, Yhat,
                                            epochs=args.epochs, lr=args.lr, dense_refeeding=args.dense_refeeding)

    t, = plt.plot([i+1 for i in range(len(train_losses))], train_losses, label="Train")
    e, = plt.plot([i+1 for i in range(len(eval_losses))], eval_losses, label="Dev")
    plt.legend(handles=[t, e])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()

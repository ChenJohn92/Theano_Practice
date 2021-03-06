import theano
from Utils import HomeDir

import pickle, numpy, gzip, os, sys

if sys.platform == "win32":
    basedir = HomeDir.GetBaseDir()
elif sys.platform == "linux":
    basedir = HomeDir.GetBaseDir()
else:
    raise LookupError("What platform is this?")

def load_mnistdata():
    """
    This is a function that used for load MNIST dataset. returns training, validation, testing dataset.

    :return:
     :return train_set, training set of the mnist dataset
     :return valid_set, validation set of the mnist dataset
     :return test_set, testing set of the mnist dataset
    """
    MNIST_dir = "Datasets\\mnist.pkl.gz"

    filename = os.path.join(basedir, MNIST_dir)

    datafile = gzip.open(filename, "rb")
    train_set, valid_set, test_set = pickle.load(datafile, encoding="latin1")

    datafile.close()
    return train_set, valid_set, test_set

if __name__ == '__main__':
    load_mnistdata()
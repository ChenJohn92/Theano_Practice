import theano
import numpy as np
import theano.tensor as T
import os

from GetStarted.LoadFile import load_mnistdata


def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, "int32")


def test_shareddata():
    trainset, validset, testset = load_mnistdata()
    trainset_x, trainset_y = shared_dataset(trainset)
    validset_x, validset_y = shared_dataset(validset)
    testset_x, testset_y = shared_dataset(testset)


if __name__ == '__main__':
    print(os.path.abspath("."))
    test_shareddata()


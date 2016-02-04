import theano
import numpy as np

import theano.tensor as T


class LogRegress(object):
    def __init__(self, X, nodes_in, nodes_out):
        """
         W is a matrix where column-k represents the separation hyperplane for class-k
            This means W output will
         b is a vector where element-k represent the free parameter of hyperplane-k
         x is a matrix where row-j represents inpu training sample-k

        :param nodes_in: the number of input. (always, the number of samples X)
        :param nodes_out: the number of output. (the number of class.)
        :X  X is the number of input samples.
        :return:
        """

        self.n_in = nodes_in
        self.n_out = nodes_out
        self.X = X

        self.W = theano.shared(value=np.zeros((self.n_in, self.n_out),
                                              dtype=theano.config.floatX),
                               name="W",
                               borrow=True)

        self.b = theano.shared(value=np.zeros((self.n_out, ),
                                              dtype=theano.config.floatX),
                               name="b",
                               borrow=True
                               )
        self.params = [self.W, self.b]
        self.Get_Predict()

    def Get_Predict(self):
        self.p_y_given_x = T.nnet.softmax(T.dot(self.X, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        # here y means a vector that gives for each example the correct lebel.

        # T.arange means that the vector contains [0, 1, 2, ..., n-1]
        # LP is the T.log(self.p_y_given_x), means the log likelihood.
        # so, LP[T.arange(y.shape[0]), y] is the vector v containing:
        # [LP[0, y[0]], LP[1, y[1]], LP[2, y[2]], ..., LP[n-1, y[n-1]] ]
        # with T.mean indicates that we want to get the mean of all the
        # log likelihood.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Error function is only a function to return a float representing
        the number of errors.

        input y should be the correct label, while y_predict should be
        the predicted result in this class.

        :param y:
        :return:
        """

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as y_pred'
            )

        if y.dtype.startswith("int"):
            return T.mean(T.neq(self.y_pred, y))

        else:
            raise NotImplementedError()






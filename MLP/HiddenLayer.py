import numpy, theano

from theano import tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input_example, n_in, n_out, W=None, b=None, activation=T.tanh):

        """
        This will generate an MLP model, units are fully-connected and have sigmoidal activation function.

        Weight W is shape (n_in, n_out), bias vector b is (n_out, )
        the unit activation can then be written as: tanh(dot(input, W) + b)

        :return:

        :param rng: a random number generator used to initialize weights.

        :param input: tensor with shape (n_examples, n_in)

        :param n_in, n_out: the input dimension and output dimension.

        :param activation: activation function.
        """

        self.input_example = input_example

        # Here it means that
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low= -numpy.sqrt(6. / (n_in + n_out)),
                    high= numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            # Activation should be 4 times larger for sigmoid than tanh.
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name="W", borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b", borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

        line_output = T.dot(input_example, self.W) + self.b

        # this is just a tricky to write the if else in oneline.
        self.output = (line_output if activation is None else activation(line_output))



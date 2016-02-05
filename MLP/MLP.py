from theano import tensor as T
from MLP.HiddenLayer import HiddenLayer
from LogisticRegress.LogRegress import LogRegress

class MLP(object):
    """
    This is the multi-layer perceptron class, this class will
    generate the multi-layer perceptron by using the hiddenlayer class
    """

    # firstly, for MLP, the input parameters are:
    # input samples (X), n_in, n_out,
    # hidden layer number: n_hidden, and that's it.

    def __init__(self, rng, input_examples, n_in, n_hidden, n_out):
        """

        :param rng: random state for initialize the weights.
        :param input_examples: input examples.
        :param n_in: number of input examples,
        :param n_hidden: number of hidden layer neurons
        :param n_out: number of output class.
        :return:
        """

        # Since we are dealing with a one hidden layer MLP, we get a hidden layer by:
        self.hiddenLayer = HiddenLayer(rng=rng, input_examples=input_examples,
                                       n_in=n_in, n_out=n_hidden, activation=T.tanh)

        self.logRegresssLayer = LogRegress(
            input_examples = self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # Here we will have the L1 and L2 norm for regularization.
        self.L1_norm = (
            abs(self.hiddenLayer.W).sum() + abs(self.logRegresssLayer.W).sum()
        )

        self.L2_norm = (
            abs(self.hiddenLayer.W).sum() + abs(self.logRegresssLayer.W).sum()
        )

        self.negative_log_likelihood = (self.logRegresssLayer.negative_log_likelihood)

        self.errors = self.logRegresssLayer.errors

        self.params = self.hiddenLayer.params + self.logRegresssLayer.params

        self.input_examples = input_examples




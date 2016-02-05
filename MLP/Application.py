from theano import tensor as T
import theano
import numpy as np
import timeit, os, pickle

from MLP.MLP import MLP
from GetStarted.LoadFile import load_mnistdata
from GetStarted.LoadShare import shared_dataset
from Utils import HomeDir


def initializeClass(rng, n_in, n_hidden, n_out):
    x = T.matrix("x")
    y = T.ivector("y")

    # we have the mnist dataset, the dataset has pictures with
    # 28 * 28=784 pixels, and classified to 10 classes (10  digits.)
    classifier = MLP(
        rng=rng,
        input_examples=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_out=n_out
    )

    return x, y, classifier

def test_mlp(learning_rate=0.01, L1_reg=0.01, L2_reg=0.0001, n_epochs=1000,
             batch_size=20, n_hidden=200):
    """
    This is a function used for testing the multi layer perceptron model.

    :param learning_rate: this is the learning rate for stochastic gradient
    :param L1_reg: this is the L1-norm weight when added to the cost
    :param L2_reg: this is the L2-norm weight when added to the cost
    :param n_epochs: this is the number of epochs to run the optimizer
    :param batch_size: this is the number of examples we will use for one iteration
    :param n_hidden: hidden layer neuron number.
    :return:
    """

    trainset, validset, testset = load_mnistdata()
    trainset_x, trainset_y = shared_dataset(trainset)
    validset_x, validset_y = shared_dataset(validset)
    testset_x, testset_y = shared_dataset(testset)

    n_train_batches = trainset_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = validset_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = testset_x.get_value(borrow=True).shape[0] / batch_size

    print("building model starts....")

    index = T.lscalar()
    x, y, MLPclassifier = initializeClass(
            rng=np.random.RandomState(6666),
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
    )

    # cost function
    cost = (MLPclassifier.negative_log_likelihood(y) + L1_reg * MLPclassifier.L1_norm +
            L2_reg * MLPclassifier.L2_norm)

    test_model = theano.function(
        inputs=[index],
        outputs=MLPclassifier.errors(y),
        givens={
            x: testset_x[index * batch_size: (index + 1) * batch_size],
            y: testset_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_model = theano.function(
        inputs=[index],
        outputs=MLPclassifier.errors(y),
        givens={
            x: validset_x[index * batch_size: (index + 1) * batch_size],
            y: validset_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    # the parameter for MLP(two W and two b)
    Gparams = [T.grad(cost, param) for param in MLPclassifier.params]

    # update function
    updates = (
        (param, param - learning_rate * Gparam)
        for param, Gparam in zip(MLPclassifier.params, Gparams)
    )


    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: trainset_x[index * batch_size: (index + 1) * batch_size],
            y: trainset_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print("now the training begins...")

    patience = 5000  # Only 5000 examples will be made.
    patience_increase = 2  # wait a little more examples.

    improvement_threshold = 0.995  # a relative improvement of threshold.

    validation_frequency = min(n_train_batches, patience / 2)
    # in this case we check every epoch.

    best_validation_loss = np.inf

    best_iter = 0

    testscore = 0

    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    # for every epoch, we will iterate the result.
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        # for each part of train batch in training dataset
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # validation_frequency is for every iteration, should we validate
            if(iter + 1) % validation_frequency == 0:
                # for every data batches in validation dataset.
                validation_losses = [valid_model(i) for i in range(n_valid_batches)]
                this_validation_losses = np.mean(validation_losses)

                print(
                    "epoch %i, minibatches %i / %i, validation error: %f %%" %
                    (epoch, minibatch_index + 1, n_train_batches, this_validation_losses * 100.)
                )
                if this_validation_losses < best_validation_loss:
                    if this_validation_losses < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_losses

                    best_iter = iter

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    testscore = np.mean(test_losses)

                    print("epoch %i, minibatch %i / %i, test error of best model %f %%",
                          (epoch, minibatch_index + 1, n_train_batches, testscore * 100.)
                    )
                    datadir = HomeDir.GetDataDir()
                    with open(os.path.join(datadir, "logRegressBestModel.pkl", "w")) as f:
                        pickle.dump(MLPclassifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print("optimization complete with best validation score of %f %%, obtained in iteration %d, with test performance %f %%" %
          (best_validation_loss * 100., best_iter + 1, testscore * 100.))

    print("the code run for %d epochs, with %f epochs/sec" % (epoch, 1. * epoch / (end_time - start_time)))

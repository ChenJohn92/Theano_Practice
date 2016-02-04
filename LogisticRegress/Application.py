from theano import tensor as T
import theano
import numpy as np
import timeit, os, pickle

from LogisticRegress.LogRegress import LogRegress
from GetStarted.LoadFile import load_mnistdata
from GetStarted.LoadShare import shared_dataset
from Utils import HomeDir

def initializeClass():
    x = T.matrix("x")
    y = T.ivector("y")

    # we have the mnist dataset, the dataset has pictures with
    # 28 * 28=784 pixels, and classified to 10 classes (10  digits.)
    classifier = LogRegress(X=x, nodes_in=28 * 28, nodes_out=10)

    return x, y, classifier


def sgd_optimization_mnist(learning_rate=0.1, n_epochs=1000,
                           batch_size=600):
    """
    This is the stochastic gradient descent training optimization part.
    :param learning_rate: learning rate used for stochastic gradient.
    :param n_epochs: maximal nnumber of epochs to run the optimizer
    :param batch_size:
    :return:
    """

    trainset, validset, testset = load_mnistdata()
    trainset_x, trainset_y = shared_dataset(trainset)
    validset_x, validset_y = shared_dataset(validset)
    testset_x, testset_y = shared_dataset(testset)

    # Here, borrow means that the value will be shallow copied, if borrow is
    # False, the value can be deep copied.
    # In this case, borrow should help to get the number of samples in train,
    # valid, test dataset. batchsize will be how many samples to compute in onetime.

    n_train_batches = trainset_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = validset_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = testset_x.get_value(borrow=True).shape[0] / batch_size

    print("building model starts....")

    # only returns a scalar variables. This means the variable only contains values.
    index = T.lscalar()

    x, y, classifier = initializeClass()

    cost = classifier.negative_log_likelihood(y)

    # Here is going to make the training, validating, testing model.
    # batch size is used for the number of samples.
    # index is which part of the sample will be used.
    # givens indicate the number of samples.
    # as a result:

    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: validset_x[index * batch_size: (index + 1) * batch_size],
            y: validset_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: testset_x[index * batch_size: (index + 1) * batch_size],
            y: testset_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # the training model is a little bit different, the parameter will
    # update parameters of the model.
    # The update is related to the W, b, learning rate, etc.
    # So, firstly, compute the gradient cost, then compute the update params

    # gradient cost:
    g_W = T.grad(cost=classifier.negative_log_likelihood(y), wrt=classifier.W)
    g_b = T.grad(cost=classifier.negative_log_likelihood(y), wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
              (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=classifier.negative_log_likelihood(y),
        updates=updates,
        givens={
            x: trainset_x[index * batch_size: (index + 1) * batch_size],
            y: trainset_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print("training the model.")

    # Training phase will train the examples.

    patience = 5000 # Only 5000 examples will be made.
    patience_increase = 2 # wait a little more examples.

    improvement_threshold = 0.995 # a relative improvement of threshold.

    validation_frequency = min(n_train_batches, patience / 2)
                                # in this case we check every epoch.

    best_validation_loss = np.inf

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

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    testscore = np.mean(test_losses)

                    print("epoch %i, minibatch %i / %i, test error of best model %f %%",
                          (epoch, minibatch_index + 1, n_train_batches, testscore * 100.)
                    )
                    datadir = HomeDir.GetDataDir()
                    with open(os.path.join(datadir, "logRegressBestModel.pkl", "w")) as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print("optimization complete with best validation score of %f %%, with test performance %f %%" %
         (best_validation_loss * 100., testscore * 100.))

    print("the code run for %d epochs, with %f epochs/sec" & (epoch, 1. * epoch / (end_time - start_time)))

# def predict():
if __name__ == '__main__':
    sgd_optimization_mnist()







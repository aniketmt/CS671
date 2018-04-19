from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from preprocess import Preprocess
import numpy as np

import torch
from torch.autograd import Variable


def train_clf(clf, n_batches=300):
    prep = Preprocess()

    # Build test dataset
    X_test, Y_test = [], []
    n_test = int(0.1 * n_batches)
    test_batches = prep.dataset[:n_test]

    # Train batch by batch
    n_batch = 0

    for batch in prep.dataset[n_test:n_batches]:
        X_train, Y_train = prep.get_data(batch)
        if X_train.shape[0] == 0:
            # print('Empty train set found')
            continue

        n_batch += 1

        clf.partial_fit(X_train, Y_train, classes=np.arange(prep.n_classes))

        # Score every 20th batch
        if n_batch % 20 == 0:
            train_score = 0.0
            for train_batch in prep.dataset[n_test:n_test + n_batch]:
                x_train, y_train = prep.get_data(train_batch)
                if x_train.shape[0] == 0:
                    continue
                train_score += clf.score(x_train, y_train)
            train_score /= n_batch

            test_score = 0.0
            for test_batch in test_batches:
                X_test, Y_test = prep.get_data(test_batch)
                if X_test.shape[0] == 0:
                    continue
                test_score += clf.score(X_test, Y_test)
            test_score /= len(test_batches)

            print('------------------------------------')
            print('batches: ', n_batch)
            print('train score: ', train_score)
            print('test score: ', test_score)
            print('------------------------------------\n')


def train_sgd(n_batches):
    sgd = linear_model.SGDClassifier()
    train_clf(clf=sgd, n_batches=n_batches)


def train_nn(n_batches):
    nn = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(50), random_state=1, learning_rate_init=0.01)
    train_clf(clf=nn, n_batches=n_batches)


def train_model(n_batches):
    return


# train_svm(300)
train_sgd(400)

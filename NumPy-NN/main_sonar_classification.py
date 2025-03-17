#!/usr/bin/env python3
"""
Created on Sun Mar 16 22:17:31 2025

@author: Dilaksan Thillaithevan

Setting up classification for CIFAR-10
"""
import numpy as np
from nn import NN
from optimiser import OPTIMIZERS
from loss import BinaryCrossEntropy
import itertools
from datasets.load import Dataset, load_sonar
from utils import one_hot_encode
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# np.seterr(all='raise')


def plot_image(X):
    X = np.transpose(np.reshape(X, (3, 32, 32)), (1, 2, 0))
    plt.imshow(X)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X, Y = load_sonar()
    Y = one_hot_encode(Y.flatten(), 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.75, shuffle=True
    )

    n_batches = 1
    batch_size = int(X_train.shape[0] / n_batches)

    n_in = 60
    layer_sizes = [(n_in, 30), (30, 10), (10, 2)]
    activations = ["Sigmoid", "Sigmoid", "Softmax"]
    initialisation = "kaiming"

    nn = NN(layer_sizes, activations, initialisations=initialisation)
    loss = BinaryCrossEntropy()

    # l0 = nn.layer(0)
    # W0 = l0.linear.W

    # out = nn(X_train)

    max_iters = 10000
    loss_tol = 1e-04

    lr = 1e-03
    adam = OPTIMIZERS["Adam"](nn, lr)

    for it in range(max_iters):
        print(f"Iteration {it} out of {max_iters}", flush=True)

        # for batch in range(n_batches):
        #     s = batch*batch_size
        #     e = s + batch_size
        #     if e > X_train.shape[0]: e = X_train.shape[0]
        #     X = X_train[s:e]
        #     Y = y_train[s:e]
        Y_hat = nn(X)
        err = loss(Y_hat, Y)
        nn.backprop(loss)
        adam.step()

        # if err <= loss_tol:
        #     print("-" * 20, f"Converged after {it} iterations", "-" * 20, flush=True)
        #     break
        # else:

        Y_hat = nn(X_test)
        err_test = loss(Y_hat, y_test)
        print(f"\t----- Loss: {err_test:.2f} -------", flush=True)

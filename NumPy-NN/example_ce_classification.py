#!/usr/bin/env python3
"""
Created on Wed Feb 19 21:27:04 2025

@author: Dilaksan Thillaithevan

NN trained on classic Iris classification problem using NumPy NN
"""
import numpy as np
from nn import NN
from optimiser import OPTIMIZERS
from loss import CrossEntropy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils import one_hot_encode
import matplotlib.pyplot as plt


if __name__ == "__main__":
    max_iters = 10000
    loss_tol = 1e-04
    print_iters = 100
    batch_size = 105
    train_size = 0.7

    # =============================================================================
    # Load data
    # =============================================================================
    iris = load_iris(as_frame=False)
    X = iris["data"]
    Y = iris["target"]

    X_names = iris["feature_names"]
    Y_names = iris["target_names"]
    Y = one_hot_encode(Y, len(Y_names))
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_size, shuffle=True
    )
    # =============================================================================
    # Define NN params
    # =============================================================================
    n_in = X.shape[1]
    n_out = Y_names.shape[0]
    activation = "Sigmoid"

    layer_sizes = [(n_in, 16), (16, 8), [8, n_out]]

    activations = ["Sigmoid", "Sigmoid", "Softmax"]

    nn = NN(layer_sizes, activations)
    ce = CrossEntropy()

    lr = 0.1
    adam = OPTIMIZERS["Adam"](nn, lr)
    hist = []

    it = 0
    conv = False
    while it < max_iters and not conv:
        for batch in range(0, len(X_train), batch_size):
            end = min(batch + batch_size, len(X_train))
            X = X_train[batch:end]
            Y = Y_train[batch:end]

            Y_hat_test = nn(X_test)
            test_loss = ce(Y_hat_test, Y_test)

            if test_loss <= loss_tol:
                conv = True
                print(
                    "-" * 20, f"Converged after {it} iterations", "-" * 20, flush=True
                )
            else:
                if it % print_iters == 0:
                    print(f"Iteration {it} out of {max_iters}", flush=True)
                    print(f"\tLoss: {test_loss:.2f}", flush=True)
                Y_hat = nn(X)
                loss = ce(Y_hat, Y)
                nn.backprop(ce)
                adam.step()
                hist += [test_loss]

        it += 1

    plt.plot(hist)
    plt.grid()
    plt.show()

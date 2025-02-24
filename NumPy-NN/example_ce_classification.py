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
from utils import one_hot_encode


if __name__ == "__main__":

    # =============================================================================
    # Load data
    # =============================================================================
    iris = load_iris(as_frame=False)
    X = iris["data"]
    Y = iris["target"]

    X_names = iris["feature_names"]
    Y_names = iris["target_names"]
    Y = one_hot_encode(Y, len(Y_names))

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

    max_iters = 1000
    loss_tol = 1e-04
    print_iters = 100

    lr = 0.1
    adam = OPTIMIZERS["Adam"](nn, lr)

    for it in range(max_iters):
        Y_hat = nn(X)
        loss = ce(Y_hat, Y)
        if loss <= loss_tol:
            print("-" * 20, f"Converged after {it} iterations", "-" * 20, flush=True)
            break
        else:
            if it % print_iters == 0:
                print(f"Iteration {it} out of {max_iters}", flush=True)
                print(f"\tLoss: {loss:.2f}", flush=True)

            nn.backprop(ce)
            adam.step()

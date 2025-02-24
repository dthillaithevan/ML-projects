#!/usr/bin/env python3
"""
Created on Sun Feb 23 12:07:47 2025

@author: Dilaksan Thillaithevan

Creating XOR gate using BCE
"""

import numpy as np
from nn import NN
from optimiser import OPTIMIZERS
from loss import BinaryCrossEntropy
from utils import one_hot_encode
import itertools


if __name__ == "__main__":

    # Combinations for xor gate
    X = np.array([p for p in itertools.product([1e-05, 0.999], repeat=2)])

    # XOR output
    X_r = np.round(X)
    Y = np.logical_xor(X_r[:, 0], X_r[:, 1]).astype(int).reshape(-1, 1)
    Y = one_hot_encode(Y.flatten(), 2)
    n_in = 2
    layer_sizes = [(n_in, 3), (3, 2)]
    nn = NN(layer_sizes, ["ReLu", "Softmax"])
    ce_loss = BinaryCrossEntropy()

    max_iters = 100
    loss_tol = 1e-04

    lr = 0.1
    adam = OPTIMIZERS["Adam"](nn, lr)

    for it in range(max_iters):
        Y_hat = nn(X)
        loss = ce_loss(Y_hat, Y)
        if loss <= loss_tol:
            print("-" * 20, f"Converged after {it} iterations", "-" * 20, flush=True)
            break
        else:
            print(f"Iteration {it} out of {max_iters}", flush=True)
            print(f"\tLoss: {loss:.2f}", flush=True)

            nn.backprop(ce_loss)
            adam.step()

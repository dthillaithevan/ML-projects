#!/usr/bin/env python3
"""
Created on Wed Feb 19 21:27:04 2025

@author: Dilaksan Thillaithevan

NN trained to represent an XOR gate of two inputs using NumPy NN
"""
import numpy as np
from nn import NN
from optimiser import OPTIMIZERS
from loss import L2
import itertools


if __name__ == "__main__":

    # Combinations for xor gate
    X = np.array([p for p in itertools.product([0, 1], repeat=2)])

    # XOR output
    Y = np.logical_xor(X[:, 0], X[:, 1]).astype(int).reshape(-1, 1)

    n_in = 2
    layer_sizes = [(n_in, 3), (3, 1)]
    nn = NN(layer_sizes, ["ReLu", "Identity"])
    l2_loss = L2()

    max_iters = 100
    loss_tol = 1e-04

    lr = 0.1
    adam = OPTIMIZERS["Adam"](nn, lr)

    for it in range(max_iters):
        Y_hat = nn(X)
        loss = l2_loss(Y_hat, Y)
        if loss <= loss_tol:
            print("-" * 20, f"Converged after {it} iterations", "-" * 20, flush=True)
            break
        else:
            print(f"Iteration {it} out of {max_iters}", flush=True)
            print(f"\tLoss: {loss:.2f}", flush=True)

            nn.backprop(l2_loss)
            adam.step()

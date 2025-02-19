#!/usr/bin/env python3
"""
Created on Tue Feb 18 10:54:53 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.
"""
from loss import L2
from nn import NN
import numpy as np

if __name__ == "__main__":

    n_in = 1
    samples = 2
    X = np.ones((samples, n_in))
    layer_sizes = [(n_in, 3), (3, 1)]
    nn = NN(layer_sizes)
    l2_loss = L2()

#     # output = nn(X)
#     # final_activation = nn.activations[-1]
#     # grad_output = nn.backprop(l2_loss)
# #
#     _, grads = nn.run_finite_difference(X, l2_loss)

# Here, we create a

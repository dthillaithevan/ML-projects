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
from datasets.load import Dataset
from utils import one_hot_encode
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_image(X):
    X = np.transpose(np.reshape(X, (3, 32, 32)), (1, 2, 0))
    plt.imshow(X)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    data = Dataset("CIFAR-10")

    n_batches = 1
    X = data.train_data / 255
    Y = data.train_labels

    ids = np.where(Y.flatten() == 0)[0]
    ids = np.append(ids, np.where(Y.flatten() == 1)[0])
    Y = Y[ids]
    Y = one_hot_encode(Y.flatten(), 2)
    X = X[ids]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.75, shuffle=True
    )

    batch_size = int(X_train.shape[0] / n_batches)

    n_in = 3072
    layer_sizes = [
        (n_in, 1024),
        (1024, 512),
        (512, 256),
        (256, 128),
        (128, 64),
        (64, 32),
        (32, 16),
        (16, 8),
        (8, 4),
        (4, 2),
    ]
    activations = ["Sigmoid"] * (len(layer_sizes) - 1) + ["Softmax"]
    initialisation = "kaiming"

    nn = NN(layer_sizes, activations, initialisations=initialisation)
    loss = BinaryCrossEntropy()

    # loss = CrossEntropy(apply_softmax=True)

    max_iters = 500
    loss_tol = 1e-04

    lr = 1e-04
    adam = OPTIMIZERS["Adam"](nn, lr)

    for it in range(max_iters):
        print(f"Iteration {it} out of {max_iters}", flush=True)

        for batch in range(n_batches):
            s = batch * batch_size
            e = s + batch_size
            if e > X_train.shape[0]:
                e = X_train.shape[0]
            X = X_train[s:e]
            Y = y_train[s:e]
            Y_hat = nn(X)
            err = loss(Y_hat, Y)
            nn.backprop(loss)
            adam.step()

        Y_hat = nn(X_test)
        err_test = loss(Y_hat, y_test)
        print(f"\t----- Loss: {err_test:.2f} -------", flush=True)

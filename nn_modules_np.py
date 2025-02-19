#!/usr/bin/env python3
"""
Created on Wed Feb 19 10:07:02 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.

This file defines modules that make up a Neural Network
"""

import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    """Abstract NN module"""

    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backprop(self, grad_out: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """Linear layer"""

    def __init__(self, n_in: int, n_out: int, random_seed: int = 1234) -> None:
        self.n_in = n_in
        self.n_out = n_out

        # Set seed
        np.random.seed(random_seed)

        # Weights
        self.W = np.random.normal(size=(n_in, n_out)) * np.sqrt(2.0 / n_in)

        # Biases
        self.b = np.zeros((n_out,), dtype=np.float32)

        # Placeholder for input & gradients
        self.x = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """z = Wx + b"""
        self.x = x
        return np.dot(x, self.W) + self.b

    def backprop(self, grad_z_output: np.ndarray) -> np.ndarray:
        """
        dL/dz^(l) = dL/dz^(l+1) W.T
        dL/dW^(l) = x.T dL/dz^(l+1)
        dL/db^(l) = sum(dL/dz^(l+1)) over batch dim

        grad_z_output = (batch, 1, n^(l+1))

        """
        # print (self.x.T.shape, grad_z_output.shape, flush = True)
        self.grad_W = np.dot(self.x.T, grad_z_output)
        self.grad_b = np.sum(grad_z_output, axis=0)
        grad_z = np.dot(grad_z_output, self.W.T)

        return grad_z


class Relu(Module):
    """Relu activation layer: a = max(0, z)"""

    def __init__(self, n_in: int) -> None:
        self.n_in = n_in

        # Placeholder for input & gradients
        self.x = None
        self.grad_a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """a = max(0, x)"""
        self.x = x
        self.a = np.maximum(0, x)
        return self.a

    def backprop(self, grad_output: np.ndarray) -> np.ndarray:
        """ """
        self.grad_a = (self.x > 0).astype(self.x.dtype) * grad_output
        return self.grad_a


class Layer(Module):
    """Single Layer: Linear + activation"""

    def __init__(self, n_in: int, n_out: int, activation: str = "ReLu"):
        assert activation in ["ReLu"]
        seed = 123

        self.linear = Linear(n_in, n_out, random_seed=seed)
        self.relu = Relu(n_out)

        self.z = None
        self.a = None
        self.grad_z = None
        self.grad_a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x -> linear(x) -> z -> Relu(z) -> a"""
        self.z = self.linear(x)
        self.a = self.relu(self.z)

        return self.a

    def backprop(self, grad_next: np.ndarray) -> np.ndarray:
        """
        dL/dz^(l) = dL/dz^(l+1) dz^(l+1)/da^(l) da^(l)/dz^(l)
        """
        self.grad_a = self.relu.backprop(grad_next)
        self.grad_z = self.linear.backprop(self.grad_a)
        return self.grad_z

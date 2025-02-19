#!/usr/bin/env python3
"""
Created on Wed Feb 19 10:07:02 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.

This file defines modules that make up a Neural Network
"""
from activations import ACTIVATIONS
from nn_module import Module
import numpy as np


class Linear(Module):
    """Linear layer"""

    def __init__(
        self, n_in: int, n_out: int, random_seed: int = 1234, name: str = None
    ) -> None:

        super().__init__(name)

        self.n_in = n_in
        self.n_out = n_out

        # Set seed
        np.random.seed(random_seed)

        # Weights
        W = np.random.normal(size=(n_in, n_out)) * np.sqrt(2.0 / n_in)
        # Biases
        b = np.zeros((n_out,), dtype=np.float32)

        self.add_params(
            ["W", "b", "x", "grad_W", "grad_b"],
            [W, b, None, None, None],
            [True, True, False, False, False],
        )

        self.num_learnable_params = 2

        # Placeholder for input & gradients

    def forward(self, x: np.ndarray) -> np.ndarray:
        """z = Wx + b"""
        self.add_param("x", x)
        return np.dot(x, self.W) + self.b

    def backprop(self, grad_z_output: np.ndarray) -> np.ndarray:
        """
        dL/dz^(l) = dL/dz^(l+1) W.T
        dL/dW^(l) = x.T dL/dz^(l+1)
        dL/db^(l) = sum(dL/dz^(l+1)) over batch dim

        grad_z_output = (batch, 1, n^(l+1))

        """
        # print (self.x.T.shape, grad_z_output.shape, flush = True)
        grad_W = np.dot(self.x.T, grad_z_output)
        grad_b = np.sum(grad_z_output, axis=0)
        grad_z = np.dot(grad_z_output, self.W.T)

        self.add_params(
            ["grad_W", "grad_b"],
            [grad_W, grad_b],
        )

        return grad_z

    @property
    def W(
        self,
    ) -> np.ndarray:
        return self.get_param("W")

    @property
    def b(
        self,
    ) -> np.ndarray:
        return self.get_param("b")

    @property
    def x(
        self,
    ) -> np.ndarray:
        return self.get_param("x")

    @property
    def grad_W(
        self,
    ) -> np.ndarray:
        return self.get_param("grad_W")

    @property
    def grad_b(
        self,
    ) -> np.ndarray:
        return self.get_param("grad_b")


class Layer(Module):
    """Single Layer: Linear + activation"""

    def __init__(
        self, n_in: int, n_out: int, activation: str = "ReLu", name: str = "layer"
    ):
        super().__init__(name)
        assert activation in ACTIVATIONS

        seed = 123

        self.linear = Linear(n_in, n_out, random_seed=seed, name=name + "_linear")

        self.activation = ACTIVATIONS[activation]()

        self.add_params(["a", "z"], [None, None])
        self.add_params(["grad_a", "grad_z"], [None, None])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x -> linear(x) -> z -> Relu(z) -> a"""
        z = self.linear(x)
        a = self.activation(z)

        self.update_params(["z", "a"], [z, a])

        return a

    def backprop(self, grad_next: np.ndarray) -> np.ndarray:
        """
        dL/dz^(l) = dL/dz^(l+1) dz^(l+1)/da^(l) da^(l)/dz^(l)
        """
        grad_a = self.activation.backprop(grad_next)
        grad_z = self.linear.backprop(grad_a)
        self.update_params(["grad_z", "grad_a"], [grad_z, grad_a])

        return grad_z

    @property
    def get_parameters(
        self,
    ) -> dict:
        return self._params

    @property
    def z(
        self,
    ) -> np.ndarray:
        """Output of linear part z = (Wx + b)"""
        return self.get_param("z")

    @property
    def a(
        self,
    ) -> np.ndarray:
        """Output of activation a = g(z) where g(x) = activation function"""
        return self.get_param("a")

    @property
    def grad_z(
        self,
    ) -> np.ndarray:
        """Gradient of linear function z w.r.t output"""
        return self.get_param("grad_z")

    @property
    def grad_a(
        self,
    ) -> np.ndarray:
        """Gradient of activation, a, w.r.t output"""
        return self.get_param("grad_a")

#!/usr/bin/env python3
"""
Created on Wed Feb 19 10:07:48 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.

This file defines a Neural Network.
"""

import numpy as np
from nn_modules_np import Layer, Linear
from loss_np import Loss


class NN:
    """Generic Linear Neural Network module"""

    # TODO: Add layer types argument (list of strs which defines the type of each layer)
    def __init__(self, layer_sizes: list[tuple]):

        # Check output size == input size for each layer
        assert all(
            [s[1] == s_n[0] for s, s_n in zip(layer_sizes[0:-1], layer_sizes[1:])]
        )

        # Number of layers
        self.N = len(layer_sizes)
        self._init_NN(layer_sizes)

        self.grad_W = [None for i in range(self.N)]
        self.grad_b = [None for i in range(self.N)]

        self.activations = [None for i in range(self.N)]
        self.W = [None for i in range(self.N)]
        self.b = [None for i in range(self.N)]

        # Whether gradients have been computed
        self.grads_exist = False

    def _init_NN(self, layer_sizes: list[tuple]) -> None:
        self.model = {f"layer_{i}": Layer(*layer_sizes[i]) for i in range(self.N)}
        self.model_size = {f"layer_{i}": layer_sizes[i] for i in range(self.N)}

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(self.N):
            x = self.layer(i)(x)

        self.W = [self.layer(i).linear.W for i in range(self.N)]
        self.b = [self.layer(i).linear.b for i in range(self.N)]
        self.activations = [self.layer(i).a for i in range(self.N)]

        return x

    def backprop(self, grad_loss: np.ndarray) -> tuple[np.ndarray]:

        # grad_output = self.grad_loss(x)
        grad_output = grad_loss

        for i in reversed(range(self.N)):
            # print(f"Computing gradient for layer {i}", flush = True)
            # print (f"{grad_output.shape=}", flush = True)
            grad_output = self.layer(i).backprop(grad_output)

        self.grad_W = [self.layer(i).linear.grad_W for i in range(self.N)]
        self.grad_b = [self.layer(i).linear.grad_b for i in range(self.N)]

        return grad_output

    @property
    def get_learnable_params(
        self,
    ) -> list[np.ndarray]:
        """Returns all learnable parameters in NN. These are the weights
        and biases of each layer. Outputs single list of Weights and biases stacked"""
        return self.W + self.b

    @property
    def get_param_gradients(
        self,
    ) -> list[np.ndarray]:
        """Returns gradients of all learnable parameters in NN. These are the weights
        and biases of each layer."""
        assert (
            self.grads_exist
        ), "Gradients have not been computed yet. Run backprop(x) first!"

        return self.grad_W + self.grad_b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def layer(self, i: int) -> Linear:
        return self.model[f"layer_{i}"]

    def layer_size(self, i: int) -> Linear:
        return self.model_size[f"layer_{i}"]

    def run_finite_difference(self, x: np.ndarray, loss: Loss) -> bool:
        """Runs finite differencing check to ensure gradients are computed
        corrected for weights (assumes bias will be correct if grad weight are correct )
        """

        # =============================================================================
        # Compute gradients analytically
        # =============================================================================
        output = self.forward(x)
        grad_loss = loss.grad(output)
        self.backprop(grad_loss)

        # Analytic gradients
        analytic_grad_weight = self.grad_W
        # analytic_grad_bias = self.grad_b

        # =============================================================================
        # Compute gradients numerically using central difference
        # =============================================================================
        epsilon = 1e-5  # dh (central diff perturbation)

        numerical_grad_weight = []
        numerical_grad_bias = []

        # Iterate over layers
        for l in range(self.N):
            layer = self.layer(l)
            numerical_grad_weight += [np.zeros_like(layer.linear.W)]
            numerical_grad_bias += [np.zeros_like(layer.linear.b)]

            # Iterate over weights
            for i in range(layer.linear.W.shape[0]):
                for j in range(layer.linear.W.shape[1]):
                    orig_value = layer.linear.W[i, j]

                    # Compute loss for (W[i, j] + epsilon)
                    layer.linear.W[i, j] = orig_value + epsilon
                    out_plus = self.forward(x)
                    loss_plus = loss(out_plus)

                    # Compute loss for (W[i, j] - epsilon)
                    layer.linear.W[i, j] = orig_value - epsilon
                    out_minus = self.forward(x)
                    loss_minus = loss(out_minus)

                    layer.linear.W[i, j] = orig_value

                    # Compute the numerical gradient (central difference)
                    numerical_grad_weight[l][i, j] = (loss_plus - loss_minus) / (
                        2 * epsilon
                    )

        # Verify gradients match
        for l in range(self.N):
            # print (numerical_grad_weight[l])
            # print (analytic_grad_weight[l])
            assert np.all(np.isclose(numerical_grad_weight[l], analytic_grad_weight[l]))

        return True

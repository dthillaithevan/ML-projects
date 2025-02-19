#!/usr/bin/env python3
"""
Created on Wed Feb 19 10:07:48 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.

This file defines a Neural Network.
"""

import numpy as np
from nn_modules import Layer, Linear, Module
from loss import Loss


class NN(Module):
    """Generic Linear Neural Network module"""

    # TODO: Add layer types argument (list of strs which defines the type of each layer)
    def __init__(
        self,
        layer_sizes: list[tuple],
        activations: str | list[str] = None,
    ):
        super().__init__()

        # Check output size == input size for each layer
        assert all(
            [s[1] == s_n[0] for s, s_n in zip(layer_sizes[0:-1], layer_sizes[1:])]
        )

        # Number of layers
        self.N = len(layer_sizes)

        if activations is None:
            activations = [None] * self.N
        else:
            if isinstance(activations, list):
                assert len(activations) == self.N
            else:
                activations = [activations] * self.N

        self._init_NN(layer_sizes, activations)

        # Keep track of whether gradients exist
        self._grads_exist = False

        # self.num_learnable_params = self.get_num_learnable_params

    def _init_NN(self, layer_sizes: list[tuple], activations: list[str]) -> None:
        self.model = {}
        self.model_size = {}
        # self.activation_names = activations
        for i in range(self.N):
            self.model[f"layer_{i}"] = Layer(
                *layer_sizes[i], name=f"layer_{i}", activation=activations[i]
            )
            self.model_size[f"layer_{i}"] = layer_sizes[i]

        # # Final layer is without activation (identity)
        # i += 1
        # self.model[f"layer_{i}"] =  Layer(*layer_sizes[i], activation = 'Identity', name = f'layer_{i}')
        # self.model_size[f"layer_{i}"] = layer_sizes[i]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(self.N):
            x = self.layer(i)(x)

        return x

    def backprop(self, loss: Loss) -> tuple[np.ndarray]:
        # Compute gradient of loss wrt last layer of activations
        grad_output = loss.grad(self.activations[-1])

        for i in reversed(range(self.N)):
            grad_output = self.layer(i).backprop(grad_output)

        self._grads_exist = True

        return grad_output

    @property
    def W(
        self,
    ) -> dict:
        return {f"W_{i}": self.layer(i).linear.W for i in range(self.N)}

    @property
    def grad_W(
        self,
    ) -> dict:
        return {f"W_{i}": self.layer(i).linear.grad_W for i in range(self.N)}

    @property
    def b(
        self,
    ) -> dict:
        return {f"b_{i}": self.layer(i).linear.b for i in range(self.N)}

    @property
    def grad_b(
        self,
    ) -> dict:
        return {f"b_{i}": self.layer(i).linear.grad_b for i in range(self.N)}

    @property
    def activations(
        self,
    ) -> list[np.ndarray]:
        return [self.layer(i).a for i in range(self.N)]  # Same as x

    def update_weights(self, new_params: list[np.ndarray]) -> None:
        current_params = self.get_learnable_params

        # Check parameters match
        assert all([True if k in new_params else False for k in current_params.keys()])

        # Propagate new weights into layers
        # Iterate over layers
        for l in range(self.N):
            # Update weights
            self.layer(l).linear.update_params(
                ["W", "b"], [new_params[f"W_{l}"], new_params[f"b_{l}"]]
            )
            self.layer(l).linear.update_params(["grad_W", "grad_b"], [None, None])

    @property
    def get_learnable_params(
        self,
    ) -> dict:
        """Returns all learnable parameters in NN. These are the weights
        and biases of each layer. Outputs dict of weights and biases labelled by each layer
        """

        return self.W | self.b

    @property
    def get_learnable_param_names(
        self,
    ) -> dict:
        """Returns all learnable parameters in NN. These are the weights
        and biases of each layer. Outputs dict of weights and biases labelled by each layer
        """

        names = []
        for i in range(self.N):
            names += [self.layer(i).linear.get_learnable_params]
        return names

    # @property
    # def get_num

    @property
    def get_param_gradients(
        self,
    ) -> dict:
        """Returns gradients of all learnable parameters in NN. These are the weights
        and biases of each layer. Outputs dict of weight and bias gradients labelled by each layer
        """
        assert (
            self._grads_exist
        ), "Gradients have not been computed yet. Run backprop(x) first!"

        return self.grad_W | self.grad_b

    def layer(self, i: int) -> Linear:
        return self.model[f"layer_{i}"]

    def layer_size(self, i: int) -> Linear:
        return self.model_size[f"layer_{i}"]

    def run_finite_difference(self, x: np.ndarray, loss: Loss, y) -> bool:
        """Runs finite differencing check to ensure gradients are computed
        corrected for weights (assumes bias will be correct if grad weight are correct )
        """
        # =============================================================================
        # Compute gradients analytically
        # =============================================================================
        # Run forard model
        self.forward(x)

        # Run gradient
        loss(x, y)
        self.backprop(loss)

        # Analytic gradients
        analytic_grad_weight = self.grad_W
        # analytic_grad_bias = self.grad_b

        # =============================================================================
        # Compute gradients numerically using central difference
        # =============================================================================
        epsilon = 1e-5  # dh (central diff perturbation)

        numerical_grad_weight = {}
        # numerical_grad_bias = {}

        # Iterate over layers
        for l in range(self.N):
            layer = self.layer(l)
            numerical_grad_weight[f"W_{l}"] = np.zeros_like(layer.linear.W)
            # numerical_grad_bias += [np.zeros_like(layer.linear.b)]

            # Iterate over weights
            for i in range(layer.linear.W.shape[0]):
                for j in range(layer.linear.W.shape[1]):
                    print(layer.linear.W, flush=True)
                    W_orig = layer.linear.W.copy()
                    W_plus = W_orig.copy()
                    W_min = W_orig.copy()

                    W_plus[i, j] += epsilon
                    W_min[i, j] -= epsilon
                    # orig_value = layer.linear.W[i, j]

                    # Compute loss for (W[i, j] + epsilon)
                    # layer.linear.W[i, j] = orig_value + epsilon
                    layer.linear.update_param("W", W_plus)

                    out_plus = self.forward(x)
                    loss_plus = loss(out_plus, y)

                    # Compute loss for (W[i, j] - epsilon)
                    # layer.linear.W[i, j] = orig_value - epsilon
                    layer.linear.update_param("W", W_min)
                    out_minus = self.forward(x)
                    loss_minus = loss(out_minus, y)

                    # layer.linear.W[i, j] = orig_value
                    layer.linear.update_param("W", W_orig)

                    # Compute the numerical gradient (central difference)
                    numerical_grad_weight[f"W_{l}"][i, j] = (loss_plus - loss_minus) / (
                        2 * epsilon
                    )

        # Verify gradients match
        for l in range(self.N):
            print(numerical_grad_weight[f"W_{l}"])
            print(analytic_grad_weight[f"W_{l}"])
            assert np.all(
                np.isclose(
                    numerical_grad_weight[f"W_{l}"], analytic_grad_weight[f"W_{l}"]
                )
            ), f"Layer {l} gradients do not match"

        return True, (numerical_grad_weight, analytic_grad_weight)

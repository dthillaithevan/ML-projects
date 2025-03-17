#!/usr/bin/env python3
"""
Created on Wed Feb 19 17:14:18 2025

@author: Dilaksan Thillaithevan
"""
import numpy as np
from nn_module import Module


class Relu(Module):
    """Relu activation layer: a = max(0, z)"""

    def __init__(
        self,
    ) -> None:
        super().__init__(name="ReLu")

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


class Identity(Module):
    """Identity activation: a = x"""

    def __init__(
        self,
    ) -> None:
        super().__init__(name="Identity")

        # Placeholder for input & gradients
        self.x = None
        self.grad_a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """a = x"""
        self.x = x
        self.a = x
        return self.a

    def backprop(self, grad_output: np.ndarray) -> np.ndarray:
        """grad = Identity"""
        # self.grad_a = np.ones_like(grad_output, dtype = self.x.dtype)
        return grad_output


class Sigmoid(Module):
    """Sigmoid activation layer: a = 1/(1 + e^-z)"""

    def __init__(
        self,
    ) -> None:
        super().__init__(name="Sigmoid")

        # Placeholder for input & gradients
        self.x = None
        self.grad_a = None

    # Avoiding overflow, adapted from: https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
    def _pos_sig(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _neg_sigmoid(self, x: np.ndarray):
        exp = np.exp(x)
        return exp / (exp + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """a = 1/(1 + e^-x)"""
        self.x = x

        self._pos = self.x >= 0
        self._neg = ~self._pos

        self.a = np.zeros_like(x, dtype=x.dtype)
        self.a[self._pos] = self._pos_sig(x[self._pos])
        self.a[self._neg] = self._pos_sig(x[self._neg])

        # self.a = 1 / (1 + np.exp(-x))

        return self.a

    def backprop(self, grad_output: np.ndarray) -> np.ndarray:
        """d(sig)/dx = sig(x) * (1-sig(x))"""
        self.grad_a = self.a * (1 - self.a) * grad_output
        return self.grad_a


class Softmax(Module):
    """Stable Softmax layer:
    m = max(x)
    a = (e^(x-m)/sum(e^(x-m)))"""

    def __init__(
        self,
    ) -> None:
        super().__init__(name="Softmax")
        self.x = None
        self.grad_a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """a = (e^(x)/sum(e^(x)))"""
        self.x = x
        m = np.max(x, axis=1, keepdims=True)
        exp = np.exp(x - m)
        self.a = exp / np.sum(exp, axis=1, keepdims=True)
        return self.a

    def backprop(self, grad_output: np.ndarray) -> np.ndarray:
        """d(a)/dx = a*(II - a), where II = Kroneker delta"""
        ii = np.eye(self.x.shape[1])
        jacobian = np.einsum("bi,ij->bij", self.a, ii) - np.einsum(
            "bi,bj->bij", self.a, self.a
        )
        self.grad_a = np.einsum("bij,bj->bi", jacobian, grad_output)
        return self.grad_a


DEFAULT_ACTIVATION = Relu
ACTIVATIONS = {
    "ReLu": Relu,
    None: DEFAULT_ACTIVATION,
    "Identity": Identity,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
}

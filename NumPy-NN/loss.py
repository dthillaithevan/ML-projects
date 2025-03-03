#!/usr/bin/env python3
"""
Created on Wed Feb 19 10:09:52 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.

This file defines Loss functions
"""
from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def forward(
        self,
    ):
        """Evaluates loss function"""
        pass

    @abstractmethod
    def grad(
        self,
    ):
        """Evaluates gradient of loss function"""
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class L2(Loss):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """L2-norm"""
        self.y = y
        return 0.5 * np.sum((x - y) ** 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """dL/da = a for L2 norm"""
        return x - self.y


class BinaryCrossEntropy(Loss):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        y_pred: np.ndarray,
        y: np.ndarray,
        integration_method: str = "sum",
        epsilon: float = 1e-10,
    ) -> float:
        """L = -[ y*log(y_pred) + (1-y)*log(1-y_pred) ]"""
        self.y = y
        self.integration_method = integration_method

        # Clip to avoid log(0)
        self.y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Loss function
        l = (
            -np.sum(y * np.log(self.y_pred) + (1 - y) * np.log(1 - self.y_pred), axis=1)
            / self.y_pred.shape[1]
        )

        # Apply integration
        if self.integration_method == "sum":
            return np.sum(l)
        elif self.integration_method == "mean":
            return np.mean(l)
        else:
            raise ValueError

    def grad(
        self,
        y_pred: np.ndarray,
        epsilon: float = 1e-10,
    ) -> np.ndarray:
        """dL/da = -y/y_pred + (1-y)/(1-y_pred)"""

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        grads = -(self.y / self.y_pred) + ((1 - self.y) / (1 - self.y_pred))

        grads /= y_pred.shape[1]

        if self.integration_method == "mean":
            grads /= y_pred.shape[0]

        return grads


class CrossEntropy(Loss):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        y_pred: np.ndarray,
        y: np.ndarray,
        integration_method: str = "sum",
        epsilon: float = 1e-10,
    ) -> float:
        """sum(y * log(y_pred)), note y_pred should be Softmax outputs!"""

        self.y = y
        # Clip to avoid log(0)
        self.y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        self.integration_method = integration_method

        loss = -np.sum(y * np.log(y_pred), axis=1)

        # Apply integration
        if self.integration_method == "sum":
            return np.sum(loss)
        elif self.integration_method == "mean":
            return np.mean(loss)
        else:
            raise ValueError

    def grad(self, y_pred: np.ndarray) -> np.ndarray:
        """dL/da = y_pred / y"""

        grad = -(self.y / y_pred)

        if self.integration_method == "mean":
            grad = grad / y_pred.shape[0]

        return grad


LOSS = {"L2": L2, "BCE": BinaryCrossEntropy, "CE": CrossEntropy}

# if __name__ == '__main__':

#     n = 10
#     y_pred = np.random.rand(10, 2)
#     y = np.random.randint(0, 2, size=(10, 2))


#     bce = BinaryCrossEntropy()

#     l = bce(y_pred, y, 'mean')
#     g = bce.grad(y_pred)

#     delta = 1e-04
#     y_pred[0,0] += delta


#     l2 = bce(y_pred, y, 'mean')

#     print (g[0,0], (l2-l)/delta)

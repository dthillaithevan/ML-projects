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

    def forward(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """L = -[ y*log(y_pred) + (1-y)*log(1-y_pred) ]"""
        self.y = y
        self.y_pred = y_pred
        return (
            -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred), axis=1)
            / y_pred.shape[0]
        )
        # return -np.sum(y * np.log(y_pred), axis = 1)

    def grad(
        self,
    ) -> np.ndarray:
        """dL/da = y_pred - y"""
        return -((self.y / self.y_pred) - (1 - self.y) / (1 - self.y_pred))


# class CrossEntropy(Loss):
#     def __init__(
#         self,
#     ):
#         super().__init__()

#     def forward(self, y_pred: np.ndarray, y: np.ndarray) -> float:
#         """  """
#         # self.y = y
#         # self.y_pred = y_pred
#         # return  -np.sum(y * np.log(y_pred) + (1-y)*np.log(1-y_pred), axis = 1)/y_pred.shape[0]
#         # # return -np.sum(y * np.log(y_pred), axis = 1)

#     def grad(self, ) -> np.ndarray:
#         """ """
#         # return -((self.y/self.y_pred)  - (1 - self.y)/(1-self.y_pred))


# class LogLikelihood(Loss):
#     """ Log Likelihood Lossfunction

#     L = -[ y*log(y_pred) + (1-y)*log(1-y_pred) ]

#     """
#     def __init__(
#         self,
#     ):
#         super().__init__()

#     def forward(self, y_pred: np.ndarray, y: np.ndarray) -> float:
#         """Expects y_pred.shape = (samples, predictions)"""
#         self.y = y
#         return - np.sum(y * np.log(y_pred) + (1-y)*np.log(1-y_pred), axis = 1)/y_pred.shape[0]

#     def grad(self, x: np.ndarray) -> np.ndarray:
#         """dL/da = a for L2 norm"""
#         return x - self.y

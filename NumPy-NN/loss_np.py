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

    def forward(self, x: np.ndarray) -> float:
        """L2-norm"""
        return 0.5 * np.sum(x**2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """dL/da = a for L2 norm"""
        return x

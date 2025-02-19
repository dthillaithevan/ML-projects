#!/usr/bin/env python3
"""
Created on Wed Feb 19 21:35:14 2025

@author: Dilaksan Thillaithevan
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class Module(ABC):
    """Abstract NN module"""

    def __init__(self, name: str = None):
        self._params = {}
        self._params_learnable = {}
        self.name = name

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backprop(self, grad_out: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def add_param(self, key: str, value: Any, learnable: bool = False) -> None:
        self._params[key] = value
        self._params_learnable[key] = learnable

    @property
    def get_params(
        self,
    ) -> dict:
        return self._params

    def add_params(
        self, keys: list[str], values: list[Any], learnable: list[bool] = None
    ) -> None:
        assert len(keys) == len(values)
        if learnable is not None:
            [self.add_param(k, v, l) for k, v, l in zip(keys, values, learnable)]
        else:
            [self.add_param(k, v) for k, v in zip(keys, values)]

    def update_param(self, key: str, new_value: Any) -> None:
        self._params.update({key: new_value})

    def update_params(self, keys: list[str], new_values: list[Any]) -> None:
        [self.update_param(k, v) for k, v in zip(keys, new_values)]

    def get_param(self, key: str) -> Any:
        return self._params.get(key, None)

    @property
    def get_learnable_params(
        self,
    ) -> list:
        return [k for k, v in self._params_learnable.items() if v]

    @property
    def get_param_names(
        self,
    ) -> list[str]:
        return list(self._params.keys())

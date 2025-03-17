#!/usr/bin/env python3
"""
Created on Mon Mar 17 16:07:12 2025

@author: Dilaksan Thillaithevan
"""

from math import sqrt
import numpy as np


def xavier(n_in: int, n_out) -> np.ndarray:
    """Xavier initialisation.
    W = U[-sqrt(6)/(sqrt(in + out)), sqrt(6)/(sqrt(in + out))]
    """
    sqrt_6 = sqrt(6)
    sqrt_inp = sqrt(n_in + n_out)
    upper = sqrt_6 / (sqrt_inp)
    lower = -upper
    out = np.random.uniform(lower, upper, (n_in, n_out))

    return out


def kaiming(n_in: int, n_out: int) -> np.ndarray:
    bound = sqrt(3 / n_in)
    return np.random.uniform(-bound, bound, size=(n_in, n_out))


# def he(n: int) -> np.ndarray:
#     """ He initialisation  """
#     std = sqrt(2.0 / n)
#     out = np.random.rand(n) * std
#     return out


INITIALISATIONS = {"xavier": xavier, "kaiming": kaiming}

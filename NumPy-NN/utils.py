#!/usr/bin/env python3
"""
Created on Sun Feb 23 11:14:37 2025

@author: Dilaksan Thillaithevan
"""

import numpy as np


def one_hot_encode(data: np.ndarray, num_classes=None):
    """
    Convert an array of integers to one-hot encoded format.

    """
    if num_classes is None:
        num_classes = np.max(data) + 1

    one_hot = np.zeros((data.size, num_classes))

    one_hot[np.arange(data.size), data] = 1

    return one_hot

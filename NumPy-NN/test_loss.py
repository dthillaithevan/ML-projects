#!/usr/bin/env python3
"""
Created on Mon Feb 24 11:39:04 2025

@author: Dilaksan Thillaithevan
"""

import numpy as np
import pytest
from loss import L2, BinaryCrossEntropy, CrossEntropy
from utils import one_hot_encode

# Tolerances for testing gradients
dH: float = 1e-5
RTOL: float = 1e-04
ATOL: float = 1e-06


def central_difference(
    loss_obj,
    input_array: np.ndarray,
    y: np.ndarray,
    input_name: str,
    integration_method: str = "sum",
) -> np.ndarray:
    """
    Central difference to verify gradients are correct
    """
    grad_numeric = np.zeros_like(input_array)

    it = np.nditer(input_array, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index

        # Perturbations
        x_plus = input_array.copy()
        x_minus = input_array.copy()
        x_plus[idx] += dH
        x_minus[idx] -= dH

        if input_name == "L2":
            # For L2 loss, forward takes (x, y)
            f_plus = loss_obj.forward(x_plus, y)
            f_minus = loss_obj.forward(x_minus, y)
        elif input_name in ("BCE", "CE"):
            # For BCE and CE, forward takes (y_pred, y, integration_method)
            f_plus = loss_obj.forward(x_plus, y, integration_method=integration_method)
            f_minus = loss_obj.forward(
                x_minus, y, integration_method=integration_method
            )
        else:
            raise ValueError("Unknown loss type")

        grad_numeric[idx] = (f_plus - f_minus) / (2 * dH)
        it.iternext()
    return grad_numeric


def test_l2_grad() -> None:
    """
    Tests the L2 loss gradient against a central difference approximation.
    """
    loss = L2()
    # Create a batch of 5 samples with 3 features each.
    x: np.ndarray = np.random.randn(5, 3)
    y: np.ndarray = np.random.randn(5, 3)
    # Run forward pass to store target.
    _ = loss.forward(x, y)
    # Compute analytic gradient.
    grad_analytic: np.ndarray = loss.grad(x)
    # Verify shape.
    assert grad_analytic.shape == x.shape, "Gradient shape mismatch for L2 loss."
    # Compute numerical gradient.
    grad_numeric = central_difference(loss, x, y, input_name="L2")
    # Compare numerical and analytic gradients.
    np.testing.assert_allclose(grad_numeric, grad_analytic, rtol=1e-4, atol=1e-6)


def test_bce_grad() -> None:
    """
    Tests the Binary Cross Entropy gradient using central differencing.
    """
    loss = BinaryCrossEntropy()

    y_pred = np.random.rand(4, 2)
    y = np.random.randint(0, 2, size=(4, 2))

    _ = loss.forward(y_pred, y, integration_method="sum")
    grad_analytic = loss.grad(y_pred)

    assert grad_analytic.shape == y_pred.shape, "Gradient shape mismatch for BCE loss."

    grad_numeric = central_difference(
        loss, y_pred, y, input_name="BCE", integration_method="sum"
    )
    np.testing.assert_allclose(grad_numeric, grad_analytic, rtol=RTOL, atol=ATOL)


def test_ce_grad() -> None:
    """
    Tests the Cross Entropy loss gradient using central differencing.
    """
    from activations import Softmax

    loss = CrossEntropy()
    softmax = Softmax()

    logits = np.random.randn(3, 4)
    y_pred = softmax(logits)

    # One-hot targets
    y = np.zeros_like(y_pred)
    for i in range(3):
        y[i, np.random.randint(0, 4)] = 1

    # Run forward pass.
    _ = loss.forward(y_pred, y, integration_method="sum")
    grad_analytic: np.ndarray = loss.grad(y_pred)

    # Check gradints
    assert (
        grad_analytic.shape == y_pred.shape
    ), "Gradient shape mismatch for Cross Entropy loss."

    # Central diff
    grad_numeric = central_difference(
        loss, y_pred, y, input_name="CE", integration_method="sum"
    )
    np.testing.assert_allclose(grad_numeric, grad_analytic, rtol=RTOL, atol=ATOL)

#!/usr/bin/env python3
"""
Created on Wed Feb 19 10:22:14 2025

@author: Dilaksan Thillaithevan
"""
from loss_np import L2
from nn_np import NN
import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):
    def __init__(self, model: NN):
        self.iteration = 0

        self.params = model.get_learnable_params
        self.n_layers = model.N

    @abstractmethod
    def step(
        self,
    ):
        """Run single iteration of optimisation"""
        pass


class Adam(Optimiser):
    """Implements Adam optimiser
    Combines RMS prop & Momentum

    x <- x - alpha*v/(sqrt(m + eps))

    v_ = velocity
    s_ =

    v_ = v_t / (1 - beta_2^t)
    m_ = m_t / (1 - beta_1^t)
    t = time (aka iteration)

    """

    def __init__(self, model: NN):
        super().__init__()

        self.beta_1 = 0.9  # Momentum EMA smoothing parameter
        self.beta_2 = 0.999  # Velocity EMA smoothing parameter
        self.alpha = 0.75  # Learning rate
        self.eps = 1e-08  # Tolerance to avoid div 0 numerical error

        self.velocities = []

    # def step(self, ) -> None:

    #     for param in self.params:
    #         for l in self.n_layers:

    # def apply_update(self, x, velocity):
    #     return x -= self.alpha *

    #     self.iteration += 1


if __name__ == "__main__":

    n_in = 1
    samples = 2
    X = np.ones((samples, n_in))
    layer_sizes = [(n_in, 3), (3, 1)]
    nn = NN(layer_sizes)
    l2_loss = L2()

    beta_1, beta_2 = 0.9, 0.999
    alpha = 0.75
    eps = 1e-08
    beta_1_inv, beta_2_inv = (1 - beta_1), (1 - beta_2)

    v = [0 for i in range(nn.num_learnable_params)]
    m = [0 for i in range(nn.num_learnable_params)]

    max_iter = 10
    t = 0

    for t in range(1, max_iter + 1):
        pred = nn(X)
        loss = l2_loss(pred)
        nn.backprop(l2_loss)
        params = nn.get_learnable_params
        params_list = list(params.values())
        params_grad = list(nn.get_param_gradients.values())

        # Compute momentum and RMS term
        m = [
            m_prev * beta_1 + beta_1_inv * grad for m_prev, grad in zip(m, params_grad)
        ]
        v = [
            v_prev * beta_2 + beta_2_inv * grad**2
            for v_prev, grad in zip(v, params_grad)
        ]

        # Bias-correction
        m_ = [_m / (1 - beta_1**t) for _m in m]
        v_ = [_v / (1 - beta_2**t) for _v in v]

        # Update parameters
        params_new_list = [
            p - alpha * _m / (_v + eps) ** 0.5 for p, _m, _v in zip(params_list, m_, v_)
        ]

        # Convert back to dict
        params_new = {k: params_new_list[i] for i, k in enumerate(params.keys())}

        # Apply params back to NN
        nn.update_learnable_params(params_new)

        print(loss)

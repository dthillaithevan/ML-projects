#!/usr/bin/env python3
"""
Created on Wed Feb 19 10:22:14 2025

@author: Dilaksan Thillaithevan
"""
from nn import NN
from abc import ABC, abstractmethod


class Optimiser(ABC):
    def __init__(self, model: NN):
        self.iteration = 0

        self.params = model.get_learnable_params
        self.n_layers = model.N
        self.model = model

    @abstractmethod
    def step(
        self,
    ):
        """Run single iteration of optimisation"""
        pass


class GradientDescent(Optimiser):
    """Vanilla Gradient Descent
    x <- x - alpha*grad
    """

    def __init__(self, model: NN, learning_rate: float = 0.8):

        super().__init__()

        self.alpha = learning_rate  # Learning rate

    def step(
        self,
    ):
        self.iteration += 1

        params = self.model.get_learnable_params
        params_grad = self.model.get_param_gradients
        params_new = {}

        for n in params_grad.keys():
            params_new[n] = params[n] - self.alpha * params_grad[n]

        # Apply params back to NN
        self.model.update_weights(params_new)


class Adam(Optimiser):
    """Implements Adam optimiser (RMS prop & Momentum)

    x <- x - alpha*v/(sqrt(m + eps))

    v = Momentum
    m = RMSprop

    v_t = momentum at time t
    m_t = RMSprop at time t

    Bias corrected terms
    v_ = v_t / (1 - beta_2^t)
    m_ = m_t / (1 - beta_1^t)
    t = time (aka iteration)
    """

    def __init__(
        self,
        model: NN,
        learning_rate: float = 0.8,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        super().__init__(model)

        self.beta_1 = beta_1  # Momentum EMA smoothing parameter
        self.beta_2 = beta_2  # Velocity EMA smoothing parameter
        self.beta_1_inv = 1 - self.beta_1
        self.beta_2_inv = 1 - self.beta_2
        self.alpha = learning_rate  # Learning rate
        self.eps = 1e-08  # Tolerance to avoid div 0 numerical error

        learnable_params = model.get_learnable_param_names

        self.v = {}
        self.m = {}

        for i in range(model.N):
            for n in learnable_params[i]:
                self.v[f"{n}_{i}"] = 0
                self.m[f"{n}_{i}"] = 0

        self.v_ = self.v.copy()
        self.m_ = self.m.copy()

    def step(
        self,
    ):
        self.iteration += 1

        params = self.model.get_learnable_params
        params_grad = self.model.get_param_gradients
        params_new = {}

        # Compute momentum and RMS term
        for n in self.v.keys():
            self.m[n] = self.m[n] * self.beta_1 + self.beta_1_inv * params_grad[n]
            self.v[n] = self.v[n] * self.beta_2 + self.beta_2_inv * params_grad[n] ** 2

            # Bias correction
            self.m_[n] = self.m[n] / (1 - self.beta_1**self.iteration)
            self.v_[n] = self.v[n] / (1 - self.beta_2**self.iteration)

            # Compute new parameters
            params_new[n] = params[n] - self.alpha * self.m_[n] / (
                (self.v_[n] + self.eps) ** 0.5
            )

        # Apply params back to NN
        self.model.update_weights(params_new)


OPTIMIZERS = {"Adam": Adam, "GradientDescent": GradientDescent}


# if __name__ == "__main__":

#     n_in = 2
#     n_out = 2
#     samples = 2
#     X = np.ones((samples, n_in))
#     X[0] = 0
#     Y = np.ones((samples, n_out))
#     Y[1] = 0
#     layer_sizes = [(n_in, 3), (3, n_out)]
#     nn = NN(layer_sizes)
#     l2_loss = L2()

#     # nn.run_finite_difference(X, l2_loss, Y)

#     beta_1, beta_2 = 0.9, 0.999
#     alpha = 0.5
#     adam = Adam(nn, learning_rate=alpha, beta_1 = beta_1, beta_2 = beta_2)

#     learnable_params = nn.get_learnable_param_names
#     v = {}
#     m = {}

#     for i in range(nn.N):
#         for n in learnable_params[i]:
#             v[f"{n}_{i}"] = 0
#             m[f"{n}_{i}"] = 0

#     v_ = v.copy()
#     m_ = m.copy()

#     max_iter = 100
#     t = 0

#     for t in range(1, max_iter + 1):
#         pred = nn(X)
#         loss = l2_loss(pred, Y)
#         print(loss)
#         nn.backprop(l2_loss)
#         adam.step()

# params = nn.get_learnable_params
# params_list = list(params.values())
# params_grad = nn.get_param_gradients
# params_new = {}

# # Compute momentum and RMS term
# for n in v.keys():
#     m[n] = m[n] * beta_1 + beta_1_inv * params_grad[n]
#     v[n] = v[n] * beta_2 + beta_2_inv * params_grad[n]**2

#     # Bias correction
#     m_[n] = m[n] / (1 - beta_1**t)
#     v_[n] = v[n] / (1 - beta_2**t)

#     params_new[n] = params[n] - alpha * m_[n] / ((v_[n] + eps) ** 0.5)

# # for n in params_grad.keys():
# #     params_new[n] = params[n] - alpha*params_grad[n]

# # Apply params back to NN
# nn.update_weights(params_new)

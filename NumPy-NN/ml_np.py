#!/usr/bin/env python3
"""
Created on Tue Feb 18 10:54:53 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.
"""
from loss_np import L2
from nn_np import NN
import numpy as np

# class Optimizer(ABC):
#     def __init__(self, model: NN):
#         self.iteration = 0

#         self.params = model.get_learnable_params
#         self.n_layers = model.N

#     @abstractmethod
#     def step(self, ):
#         """ Run single iteration of optimization """
#         pass


# class Adam(Optimizer):
#     def __init__(self, model: NN):
#         super().__init__()

#         self.beta_1 = 0.9
#         self.beta_2 = 0.999
#         self.alpha = 0.75
#         self.eps = 1e-08

# self.velocities =

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

    # output = nn(X)
    # final_activation = nn.activations[-1]
    # grad_output = nn.backprop(final_activation)

    nn.run_finite_difference(X, l2_loss)

    # Here, we create a test input with 4 samples and 3 features each.
    # X = np.random.randn(2, 3)

    # # -------------------------------
    # # 2. Initialize the module
    # # -------------------------------
    # # Let's use a LinearReLU module with input dimension 3 and output dimension 2.
    # module = Layer(n_in=3, n_out=1)

    # # -------------------------------
    # # 3. Forward and Backward Passes
    # # -------------------------------
    # # Forward pass: compute the output of the module.
    # output = module.forward(X)

    # # Define a simple loss: L = 0.5 * sum(output^2)
    # loss = Loss(output)

    # # The gradient of the loss with respect to the output is just 'output'.
    # grad_output = Loss_grad(output)

    # # Backward pass: compute gradients with respect to the input and parameters.
    # module.backprop(grad_output)

    # # Analytic gradients stored in the linear module:
    # analytic_grad_weight = module.linear.grad_W
    # analytic_grad_bias = module.linear.grad_b

    # # -------------------------------
    # # 4. Finite Difference Gradient Check
    # # -------------------------------
    # epsilon = 1e-5  # small perturbation for finite differences

    # # Prepare numerical gradients arrays with the same shape as the parameters.
    # numerical_grad_weight = np.zeros_like(module.linear.W)
    # numerical_grad_bias = np.zeros_like(module.linear.b)

    # # ----- Check gradients for weights -----
    # for i in range(module.linear.W.shape[0]):
    #     for j in range(module.linear.W.shape[1]):
    #         # Save the original value.
    #         orig_value = module.linear.W[i, j]

    #         # Compute loss for (W[i, j] + epsilon)
    #         module.linear.W[i, j] = orig_value + epsilon
    #         out_plus = module.forward(X)
    #         loss_plus = Loss(out_plus)

    #         # Compute loss for (W[i, j] - epsilon)
    #         module.linear.W[i, j] = orig_value - epsilon
    #         out_minus = module.forward(X)
    #         loss_minus = Loss(out_minus)

    #         # Restore original value.
    #         module.linear.W[i, j] = orig_value

    #         # Compute the numerical gradient (central difference)
    #         numerical_grad_weight[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # n_in = 3
    # n_out = 2

    # layer = Layer(n_in, n_out)

    # x = np.arange(n_in).reshape(1, n_in)

    # out = layer(x)
    # L = loss(out)

    # dLda = loss_grad(out)
    # dz = layer.backprop(dLda)

    # # dLdz =

    # a = layer.relu.a
    # grad_a = layer.relu.grad_a
    # grad_z = layer.grad_z
    # W = layer.lin.W
    # grad_W = layer.lin.grad_W
    # z = layer.z

    # z_ = layer.z.copy()
    # delta = 1e-04
    # z_[0,0] += delta
    # a = layer.relu(z_)
    # L_ = loss(a)

    # dL = (L_ - L)/delta

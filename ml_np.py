#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:54:53 2025

@author: Dilaksan Thillaithevan

Implenting NN from scratch using NumPy for my own benefit.
"""

import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    """ Abstract NN module """
    def __init__(self,):
        pass
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backprop(self, grad_out: np.ndarray) -> np.ndarray:
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    """ Linear layer """
    def __init__(self, n_in: int, n_out: int, random_seed: int = 1234) -> None:
        self.n_in = n_in
        self.n_out = n_out
        
        # Set seed
        np.random.seed(random_seed)
        
        # Weights
        self.W = np.random.normal(size=(n_in, n_out)) * np.sqrt(2. / n_in)

        # Biases
        self.b = np.zeros((n_out,), dtype = np.float32)
        
        # Placeholder for input & gradients
        self.x = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """ z = Wx + b """
        self.x = x
        return np.dot(x, self.W) + self.b

    def backprop(self, grad_z_output: np.ndarray) -> np.ndarray:
        """ 
        dL/dz^(l) = dL/dz^(l+1) W.T
        dL/dW^(l) = x.T dL/dz^(l+1)  
        dL/db^(l) = sum(dL/dz^(l+1)) over batch dim
        
        grad_z_output = (batch, 1, n^(l+1))
        
        """
        print (self.x.T.shape, grad_z_output.shape, flush = True)
        self.grad_W = np.dot(self.x.T, grad_z_output)
        self.grad_b = np.sum(grad_z_output, axis = 0)
        grad_z = np.dot(grad_z_output, self.W.T)
        
        return grad_z
    
class Relu(Module):
    """ Relu activation layer """
    def __init__(self, n_in: int) -> None:
        self.n_in = n_in
        
        # Placeholder for input & gradients
        self.x = None
        self.grad_a = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """ a = max(0, x) """
        self.x = x
        self.a = np.maximum(0, x)
        return self.a

    def backprop(self, grad_output: np.ndarray) -> np.ndarray:
        """ 
        """
        self.grad_a = (self.x > 0).astype(self.x.dtype) * grad_output
        return self.grad_a 
        
    
class Layer(Module):
    """ Single Layer: Linear + activation """
    def __init__(self, n_in: int, n_out: int, activation: str = 'ReLu'):
        assert activation in ['ReLu']
        seed = 123
        
        self.linear = Linear(n_in, n_out, random_seed=seed)
        self.relu = Relu(n_out)
        
        self.z = None
        self.a = None
        self.grad_z = None
        self.grad_a = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """ x -> linear(x) -> z -> Relu(z) -> a """
        self.z = self.linear(x)
        self.a = self.relu(self.z)
        
        return self.a
        
    def backprop(self, grad_next: np.ndarray) -> np.ndarray:
        """
        dL/dz^(l) = dL/dz^(l+1) dz^(l+1)/da^(l) da^(l)/dz^(l)
        """
        self.grad_a = self.relu.backprop(grad_next)
        self.grad_z = self.linear.backprop(self.grad_a)
        return self.grad_z


class NN:
    """ Neural Network module """
    def __init__(self, layer_sizes: list[tuple]):
        
        # Check output size == input size for each layer
        assert all([s[1] == s_n[0] for s, s_n in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
        # Number of layers
        self.N = len(layer_sizes)
        self._init_NN(layer_sizes)
        
        self.grad_W = [None for i in range(self.N)]
        self.grad_b = [None for i in range(self.N)]
        
        self.activations = [None for i in range(self.N)]
        self.W = [None for i in range(self.N)]
        self.b = [None for i in range(self.N)]
        
        # Whether gradients have been computed
        self.grads_exist = False
        
        
    def _init_NN(self, layer_sizes: list[tuple]) -> None:
        self.model = {f"layer_{i}" : Layer(*layer_sizes[i]) for i in range(self.N)}
        self.model_size = {f"layer_{i}" : layer_sizes[i] for i in range(self.N)}
    
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(self.N):
            x = self.layer(i)(x)
            
        self.W = [self.layer(i).linear.W for i in range(self.N)]
        self.b = [self.layer(i).linear.b for i in range(self.N)]
        self.activations = [self.layer(i).a for i in range(self.N)]
        
        return x

    def backprop(self, x: np.ndarray) -> tuple[np.ndarray]:
        
        grad_output = self.grad_loss(x)
        
        print (grad_output)
        for i in reversed(range(self.N)):
            print(f"Computing gradient for layer {i}", flush = True)
            print (f"{grad_output.shape=}", flush = True)
            grad_output = self.layer(i).backprop(grad_output)
        
        self.grad_W = [self.layer(i).linear.grad_W for i in range(self.N)]
        self.grad_b = [self.layer(i).linear.grad_b for i in range(self.N)]
        
        return grad_output
    
    @property
    def get_learnable_params(self, ) -> list[np.ndarray]:
        """ Returns all learnable parameters in NN. These are the weights
        and biases of each layer. Outputs single list of Weights and biases stacked """
        return self.W + self.b
    
    @property
    def get_param_gradients(self, ) -> list[np.ndarray]:
        """ Returns gradients of all learnable parameters in NN. These are the weights
        and biases of each layer. """
        assert self.grads_exist, "Gradients have not been computed yet. Run backprop(x) first!" 
        
        return self.grad_W + self.grad_b
    
    
    
    def loss(self, x: np.ndarray) -> float:
        return Loss(self.forward(x))
    
    def grad_loss(self, x: np.ndarray) -> np.ndarray:
        return Loss_grad(x)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def layer(self, i:int) -> Linear:
        return self.model[f"layer_{i}"]
        
    
    def layer_size(self, i:int) -> Linear:
        return self.model_size[f"layer_{i}"]
    
    def run_finite_difference(self, x: np.ndarray) -> bool:
        """ Runs finite differencing check to ensure gradients are computed 
        corrected for weights (assumes bias will be correct if grad weight are correct ) """
        
        # =============================================================================
        # Compute gradients analytically
        # =============================================================================
        output = self.forward(x)
        loss = Loss(output)
        grad_output = Loss_grad(output)
        self.backprop(grad_output)
        
        # Analytic gradients
        analytic_grad_weight = self.grad_W
        analytic_grad_bias = self.grad_b
        
        # =============================================================================
        # Compute gradients numerically using central difference   
        # =============================================================================
        epsilon = 1e-5  # dh (central diff perturbation)
        
        numerical_grad_weight = []
        numerical_grad_bias = []
        
        # Iterate over layers
        for l in range(self.N):
            layer = self.layer(l)
            numerical_grad_weight += [np.zeros_like(layer.linear.W)]
            numerical_grad_bias += [np.zeros_like(layer.linear.b)]
            
            # Iterate over weights
            for i in range(layer.linear.W.shape[0]):
                for j in range(layer.linear.W.shape[1]):
                    orig_value = layer.linear.W[i, j]
            
                    # Compute loss for (W[i, j] + epsilon)
                    layer.linear.W[i, j] = orig_value + epsilon
                    out_plus = self.forward(x)
                    loss_plus = Loss(out_plus)
            
                    # Compute loss for (W[i, j] - epsilon)
                    layer.linear.W[i, j] = orig_value - epsilon
                    out_minus = self.forward(x)
                    loss_minus = Loss(out_minus)
            
                    layer.linear.W[i, j] = orig_value
            
                    # Compute the numerical gradient (central difference)
                    numerical_grad_weight[l][i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
        # Verify gradients match
        for l in range(self.N):
            assert np.all(np.isclose(numerical_grad_weight[l], analytic_grad_weight[l]))
            
def Loss(x: np.ndarray) -> float:
    """ L2-norm """
    return 0.5*np.sum(x**2)

def Loss_grad(x: np.ndarray) -> np.ndarray:
    """ dL/da = a for L2 norm """
    return x

class Optimizer(ABC):
    def __init__(self, model: NN):
        self.iteration = 0
        
        self.params = model.get_learnable_params
        self.n_layers = model.N
        
    @abstractmethod
    def step(self, ):
        """ Run single iteration of optimization """
        pass
    
    
    
class Adam(Optimizer):
    def __init__(self, model: NN):
        super().__init__()
        
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.alpha = 0.75
        self.eps = 1e-08
        
        self.velocities = 
        
    def step(self, ) -> None:
        
        for param in self.params:
            for l in self.n_layers:
                
        
    def apply_update(self, x, velocity):
        return x -= self.alpha * 
        
        self.iteration += 1
        
        
        
 
if __name__ == '__main__':

    n_in = 1
    samples = 2    
    X = np.ones((samples, n_in))
    layer_sizes = [(n_in, 3), (3, 1)]
    nn = NN(layer_sizes)
    
    output = nn(X)
    # final_activation = nn.activations[-1]
    # grad_output = nn.backprop(final_activation)
    
    # fd, grad = nn.run_finite_difference(X)
    
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
    
    
    
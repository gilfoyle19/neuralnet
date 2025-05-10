import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Adjust the path to the parent directory
from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from nn.common import differntial

class Activation(differntial):

    """
    Abstract base class for activation functions in a neural network.
    __call__ method is used to apply the activation function to the input tensor.
    The activation function should be differentiable, and the derivative can be
    computed using the gradient method.
    """

    @abstractmethod
    def __call__(self, input_tensor: NDArray) -> NDArray: ...


class Sigmoid(Activation):
    """Sigmoid activation function."""
    def __call__(self, input_tensor: NDArray) -> NDArray:
        """
        Apply the sigmoid activation function to the input tensor.
        """
        return 1.0/(1 + np.exp(-1 * input_tensor))
    
    def gradient(self, input_tensor: NDArray) -> NDArray:
        """
        Compute the gradient of the sigmoid activation function.
        """
        return self(input_tensor) * (1 - self(input_tensor))   
    

class ReLU(Activation):
    """ReLU activation function."""
    def __call__(self, input_tensor: NDArray) -> NDArray:
        """
        Apply the ReLU activation function to the input tensor.
        """
        return np.maximum(0, input_tensor)
    
    def gradient(self, input_tensor: NDArray) -> NDArray:
        """
        Compute the gradient of the ReLU activation function.
        """
        _result = input_tensor.copy()
        _result[input_tensor > 0] = 1
        _result[input_tensor <= 0] = 0
        return _result

class linear(Activation):
    """Linear activation function."""
    def __call__(self, input_tensor: NDArray) -> NDArray:
        """
        Apply the linear activation function to the input tensor.
        """
        return input_tensor
    
    def gradient(self, input_tensor: NDArray) -> NDArray:
        """
        Compute the gradient of the linear activation function.
        """
        return np.ones_like(input_tensor) #ones_like(input_tensor) returns an array of ones with the same shape as input_tensor

class tanh(Activation):
    """Tanh activation function."""
    def __call__(self, input_tensor: NDArray) -> NDArray:
        """
        Apply the tanh activation function to the input tensor.
        """
        return np.tanh(input_tensor)
    
    def gradient(self, input_tensor: NDArray) -> NDArray:
        """
        Compute the gradient of the tanh activation function.
        """
        return 1 - np.tanh(input_tensor)**2

class softmax(Activation):
    """Softmax activation function."""
    def __call__(self, input_tensor: NDArray) -> NDArray:
        """
        Apply the softmax activation function to the input tensor.
        """
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

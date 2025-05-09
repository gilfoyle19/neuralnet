from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray

class Activation(Differentiable):

    @abstractmethod
    def __call__(self, input_tensor: NDArray) -> NDArray: ...


class Sigmoid(Activation):
    """Sigmoid activation function."""
    pass

class ReLU(Activation):
    """ReLU activation function."""
    pass

class linear(Activation):
    """Linear activation function."""
    pass

class tanh(Activation):
    """Tanh activation function."""
    pass

class softmax(Activation):
    """Softmax activation function."""
    pass
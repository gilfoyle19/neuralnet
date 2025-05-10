import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Adjust the path to the parent directory
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from nn.optimizer import Optimizer  # Use relative import

class Layer(ABC):
    """
    Abstract base class for layers in a neural network.
    """
    @property
    @abstractmethod
    def output(self) -> NDArray:
        """
        Returns the output of the layer.
        """
        pass

    @abstractmethod
    def __call__(self, input_tensor: NDArray) -> NDArray:
        """
        Forward pass through the layer.
        """
        pass

    @abstractmethod
    def build(self, input_tensor: NDArray):
        """
        Build the layer with the given input tensor.
        """
        pass

    @abstractmethod
    def update(self, optimizer: Optimizer):
        """
        Update the layer's parameters using the optimizer.
        """
        pass


class Dense(Layer):
    def __init__(self, units: int):
        """
        Initialize a Dense layer with the given number of units.
        """
        self._units = units  # Number of units in the layer
        self._input_units = None  # Number of input units
        self._weights = None
        self._bias = None
        self._output = None
        self._dw = None
        self._db = None

    @property
    def output(self) -> NDArray:
        """
        Returns the output of the layer.
        """
        return self._output

    @property
    def weights(self):
        """
        Returns the weights of the layer.
        """
        return self._weights

    @weights.setter
    def weights(self, weights: NDArray):
        """
        Sets the weights of the layer.
        """
        self._weights = weights

    @property
    def bias(self):
        """
        Returns the bias of the layer.
        """
        return self._bias

    @bias.setter
    def bias(self, bias: NDArray):
        """
        Sets the bias of the layer.
        """
        self._bias = bias

    @property
    def dw(self):
        """
        Returns the gradient of the weights.
        """
        return self._dw

    @dw.setter
    def dw(self, gradients: NDArray):
        """
        Sets the gradient of the weights.
        """
        self._dw = gradients

    @property
    def db(self):
        """
        Returns the gradient of the bias.
        """
        return self._db

    @db.setter
    def db(self, gradients: NDArray):
        """
        Sets the gradient of the bias.
        """
        self._db = gradients

    def build(self, input_tensor: NDArray):
        """
        Build the layer with the given input tensor.
        """
        self._input_units = input_tensor.shape[1]  # Number of input units
        self._weights = np.random.randn(self._input_units, self._units) * np.sqrt(2. / self._input_units)  # He initialization
        self._bias = np.zeros((1, self._units))  # Initialize bias to zeros

    def __call__(self, input_tensor: NDArray) -> NDArray:
        """
        Forward pass through the layer.
        """
        if self._weights is None or self._bias is None:
            self.build(input_tensor)  # Build the layer if not already built

        self._output = np.dot(input_tensor, self._weights) + self._bias  # Compute the output
        return self._output

    def update(self, optimizer: Optimizer):
        """
        Update the weights and bias using the optimizer.
        """
        optimizer.update_weights(self, self.dw)  # Call the update_weights method of the optimizer
        optimizer.update_bias(self, self.db)  # Call the update_bias method of the optimizer


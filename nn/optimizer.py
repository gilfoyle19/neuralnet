import numpy as np
from numpy.typing import NDArray

class Optimizer:
    def __init__(self, learning_rate: float):
        """
        Initialize the optimizer with a learning rate.
        """
        self._learning_rate = learning_rate 
        self._layer_number = 0 # Number of layers in the model

        @property
        def layer_number(self):
            """
            Returns the number of layers in the model.
            """
            return self._layer_number
        
        @layer_number.setter
        def layer_number(self, layer_number: int):
            """
            Sets the number of layers in the model.
            """
            self._layer_number = layer_number

        def update_weights(self, layer, grad_weights):
            """
            Update the weights of the layer using the optimizer.
            """
            layer.weights -= self._learning_rate * grad_weights
        def update_bias(self, layer, grad_bias):
            """
            Update the bias of the layer using the optimizer.
            """
            layer.bias -= self._learning_rate * grad_bias
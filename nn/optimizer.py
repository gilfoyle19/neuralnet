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

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, learning_rate: float):
        """
        Initialize the SGD optimizer with a learning rate.
        """
        super().__init__(learning_rate)
    
    def update_weights(self, layer, grad_weights):
        layer.weights -= self._learning_rate * grad_weights
    
    def update_bias(self, layer, grad_bias):
        layer.bias -= self._learning_rate * grad_bias



class Adam(Optimizer):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer with a learning rate, beta1, beta2, and epsilon.
        """
        super().__init__(learning_rate)
        self.beta_1 = beta1
        self.beta_2 = beta2
        self.epsilon = epsilon
        self.m_w = {} # Initialize first moment vector
        self.v_w = {} # Initialize second moment vector
        self.m_b = {} # Initialize first moment vector
        self.v_b = {} # Initialize second moment vector
        self.pw = 1 

    def update_weights(self, layer, grad_weights: NDArray):
        if not self._layer_number in self.m_w.keys():
            self.m_w[self._layer_number] = np.zeros_like(grad_weights) # Initialize first moment vector
            self.v_w[self._layer_number] = np.zeros_like(grad_weights) # Initialize second moment vector
        # Update the first and second moment vectors
        
        self.pw += 1 # Increment the time step
        self.m_w[self._layer_number] = self.beta_1 * self.m_w[self._layer_number] + (1 - self.beta_1) * grad_weights
        self.v_w[self._layer_number] = self.beta_2 * self.v_w[self._layer_number] + (1 - self.beta_2) * np.square(grad_weights)
        m_hat = self.m_w[self._layer_number] / (1 - np.power(self.beta_1, self.pw))
        v_hat = self.v_w[self._layer_number] / (1 - np.power(self.beta_2, self.pw))
        layer.weights -= self._learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def update_bias(self, layer, grad_bias: NDArray):
        if not self._layer_number in self.m_b.keys():
            self.m_b[self._layer_number] = np.zeros_like(grad_bias)
            self.v_b[self._layer_number] = np.zeros_like(grad_bias)
        
        self.pw += 1 
        self.m_b[self._layer_number] = self.beta_1 * self.m_b[self._layer_number] + (1 - self.beta_1) * grad_bias
        self.v_b[self._layer_number] = self.beta_2 * self.v_b[self._layer_number] + (1 - self.beta_2) * np.square(grad_bias)
        m_hat = self.m_b[self._layer_number] / (1 - np.power(self.beta_1, self.pw))
        v_hat = self.v_b[self._layer_number] / (1 - np.power(self.beta_2, self.pw))
        layer.bias -= self._learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        """
        Initialize the RMSprop optimizer with a learning rate, decay rate, and epsilon.
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache_w = {} # Initialize cache for weights
        self.cache_b = {} # Initialize cache for bias

        def update_weights(self, layer, grad_weights: NDArray):
            if not self._layer_number in self.cache_w.keys():
                self.cache_w[self._layer_number] = np.zeros_like(grad_weights)
            
            # Update the cache
            self.cache_w[self._layer_number] = self.decay_rate * self.cache_w[self._layer_number] + (1 - self.decay_rate) * np.square(grad_weights)

            # Update the weights
            layer.weights -= self._learning_rate * grad_weights / (np.sqrt(self.cache_w[self._layer_number]) + self.epsilon)

        def update_bias(self, layer, grad_bias: NDArray):
            if not self._layer_number in self.cache_b.keys():
                self.cache_b[self._layer_number] = np.zeros_like(grad_bias)
            
            # Update the cache
            self.cache_b[self._layer_number] = self.decay_rate * self.cache_b[self._layer_number] + (1 - self.decay_rate) * np.square(grad_bias)

            # Update the bias
            layer.bias -= self._learning_rate * grad_bias / (np.sqrt(self.cache_b[self._layer_number]) + self.epsilon)
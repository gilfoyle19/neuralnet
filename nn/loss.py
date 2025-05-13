from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray

from nn.common import differntial

class Loss(differntial):
    """
    Abstract base class for loss functions.
    """
    @abstractmethod
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray: ...
    """
    Calculate the loss between predictions and labels: Forward pass."""


class MeanSquaredError(Loss):
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return np.mean(np.square(predictions - labels))

    def gradient(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return 2 * (predictions - labels) / labels.size
    
class BinaryCrossEntropy(Loss):
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return -np.mean(labels * np.log(predictions + 1e-15) + (1 - labels) * np.log(1 - predictions + 1e-15))

    def gradient(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return (predictions - labels) / (predictions * (1 - predictions) + 1e-15)
    
class MeanAbsoluteError(Loss):
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return np.mean(np.abs(predictions - labels))

    def gradient(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return np.sign(predictions - labels) / labels.size
    

from abc import ABC, abstractmethod
from typing import Any

class differntial(ABC):
    """
    Abstract base class for differentials.
    """
    @abstractmethod
    def gradient(self, *args, **kwargs) -> Any: ... # An abstract method common to all differentials


from abc import ABC, abstractmethod
from typing import Any

class differntial(ABC):
    """
    Abstract base class for differentials.
    """
    @abstractmethod
    def gradient(self, *args, **kwargs) -> Any: ... # reason: this is an abstract method


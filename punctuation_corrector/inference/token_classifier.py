from abc import ABC, abstractmethod
import numpy as np


class TokenClassifier(ABC):
    @abstractmethod
    def predict(self, tokens: np.ndarray) -> np.ndarray:
        pass

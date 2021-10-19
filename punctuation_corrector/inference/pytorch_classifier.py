from transformers import AutoConfig, AutoModelForTokenClassification
import numpy as np
import torch

from .token_classifier import TokenClassifier


class PytorchClassifier(TokenClassifier):
    def __init__(self, model_path: str) -> None:
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            config=AutoConfig.from_pretrained(model_path)
        ).to(self._device)
        self._model.eval()

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tokens = torch.from_numpy(tokens[None, :]).to(self._device)
            batch_result = self._model.forward(tokens, attention_mask=tokens > 0)[0][0]
            return torch.sigmoid(batch_result).detach().cpu().numpy()

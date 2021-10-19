import numpy as np
import onnxruntime
from pathlib import Path
import gzip

from .token_classifier import TokenClassifier


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class OnnxClassifier(TokenClassifier):
    def __init__(self, model_path: str) -> None:
        onnx_model_path = self.onnx_model_path(model_path)

        with gzip.open(onnx_model_path, 'rb') as model_file:
            self._session = onnxruntime.InferenceSession(
                model_file.read(),
                onnxruntime.SessionOptions(),
                providers=['CPUExecutionProvider']
            )

    @staticmethod
    def onnx_model_path(path: str) -> Path:
        return Path(path) / 'model.onnx.gz'

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        tokens = tokens[None, :]
        ort_inputs = {
            'input_ids':  tokens,
            'attention_mask':  (tokens > 0).astype(np.int64),
            'token_type_ids':  np.zeros_like(tokens, dtype=np.int64)
        }
        return sigmoid(self._session.run(None, ort_inputs)[0][0])

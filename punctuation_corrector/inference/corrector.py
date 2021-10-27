import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Iterable, List, Tuple

from .token_classifier import TokenClassifier
from .model_cache import ModelCache
from .output_formatter import OutputFormatter, DefaultOutputFormatter
from ..common.preprocessing import encode_punctuation, parse_output


def _load_onnx_classifier() -> TokenClassifier:
    from .onnx_classifier import OnnxClassifier
    return OnnxClassifier


def _load_pytorch_classifier() -> TokenClassifier:
    from .pytorch_classifier import PytorchClassifier
    return PytorchClassifier


CLASSIFIER_LOADERS = {
    'onnx': _load_onnx_classifier,
    'onnx_quantized': _load_onnx_classifier,
    'pytorch': _load_pytorch_classifier
}


def _create_classifier(classifier_name: str) -> TokenClassifier:
    return CLASSIFIER_LOADERS[classifier_name]()


class Corrector:
    MODEL_CACHE = ModelCache(Path.home() / '.cache' / 'punctuation_corrector')
    MAX_LEN = 512
    OVERLAP = 10

    def __init__(
            self,
            token_classifier: TokenClassifier,
            tokenizer: PreTrainedTokenizerBase,
            labels: List[str],
            predict_case: bool,
            output_formatter: OutputFormatter) -> None:
        self._token_classifier = token_classifier
        self._tokenizer = tokenizer
        self._labels = labels
        self._predict_case = predict_case
        self._output_formatter = output_formatter

    @staticmethod
    def _create_splits(text_len: int) -> List[Tuple[int, int, int, int]]:
        splits = []

        input_start = 0
        input_end = 0
        while input_end < text_len:
            input_end = min(input_start + Corrector.MAX_LEN, text_len)
            result_start = 0
            if input_start > 0:
                result_start += Corrector.OVERLAP

            result_end = input_end - input_start
            if input_end < text_len:
                result_end -= Corrector.OVERLAP

            splits.append((input_start, input_end, result_start, result_end))
            input_start += Corrector.MAX_LEN - 2 * Corrector.OVERLAP

        return splits

    def correct(self, texts: Iterable[str]) -> List[str]:
        results = []
        for text in texts:
            encoded_text = encode_punctuation(
                self._tokenizer,
                self._tokenizer.convert_tokens_to_ids(self._labels),
                text)

            predicted_labels = []
            predicted_case = []
            splits = self._create_splits(len(encoded_text.tokens))
            for input_start, input_end, result_start, result_end in splits:
                result = self._token_classifier.predict(encoded_text.tokens[input_start:input_end])
                result = parse_output(result[result_start:result_end], self._predict_case)
                predicted_labels.append(result.labels)
                predicted_case.append(result.case_labels)

            predicted_labels = np.concatenate(predicted_labels, axis=0)
            if self._predict_case:
                predicted_case = np.concatenate(predicted_case, axis=0)
            else:
                predicted_case = [None] * len(encoded_text.tokens)

            previous_span_end = 0
            corrected_spans = []
            spans = zip(
                predicted_labels,
                predicted_case,
                encoded_text.labels,
                encoded_text.start_offsets,
                encoded_text.end_offsets)

            for predicted_labels, predicted_case, original_labels, span_start, span_end in spans:
                span_start = max(span_start, previous_span_end)
                span_end = max(span_end, previous_span_end)

                original_punctuation = text[previous_span_end:span_start]
                original_span = text[span_start:span_end]
                corrected_span = self._output_formatter.format(
                    self._labels,
                    predicted_labels,
                    predicted_case,
                    original_punctuation,
                    original_span,
                    original_labels)

                corrected_spans.append(corrected_span)
                previous_span_end = span_end

            results.append(''.join(corrected_spans))

        return results

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path / 'punctuation_corrector.json'

    @classmethod
    def save_metadata(
            cls,
            class_name: str,
            labels: List[str],
            predict_case: bool,
            path: Path) -> None:

        metadata = {'class': class_name, 'labels': labels, 'predict_case': predict_case}
        with open(cls._metadata_path(path), 'w') as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, path: str, output_formatter: OutputFormatter = DefaultOutputFormatter()):
        cached_path = cls.MODEL_CACHE.cached_path(path)

        with open(cls._metadata_path(cached_path)) as f:
            config = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(
            cached_path,
            do_lower_case=False
        )

        return cls(
            _create_classifier(config['class'])(str(cached_path)),
            tokenizer,
            config.get('labels', []),
            config.get('predict_case', False),
            output_formatter
        )

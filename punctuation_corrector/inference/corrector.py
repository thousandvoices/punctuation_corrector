import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Iterable, List

from .token_classifier import TokenClassifier
from .model_cache import ModelCache
from .output_formatter import OutputFormatter, DefaultOutputFormatter
from ..common.preprocessing import encode_punctuation


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

    def __init__(
            self,
            token_classifier: TokenClassifier,
            tokenizer: PreTrainedTokenizerBase,
            labels: List[str],
            output_formatter: OutputFormatter) -> None:
        self._token_classifier = token_classifier
        self._tokenizer = tokenizer
        self._labels = labels
        self._output_formatter = output_formatter

    def correct(self, texts: Iterable[str]) -> List[str]:
        results = []
        for text in texts:
            encoded_text = encode_punctuation(
                self._tokenizer,
                self._tokenizer.convert_tokens_to_ids(self._labels),
                text,
                512)
            predicted_labels = self._token_classifier.predict(encoded_text.tokens)

            previous_span_end = 0
            corrected_spans = []
            spans = zip(
                predicted_labels,
                encoded_text.labels,
                encoded_text.start_offsets,
                encoded_text.end_offsets)

            for predicted_labels, original_labels, start_offset, end_offset in spans:
                start_offset = max(start_offset, previous_span_end)
                end_offset = max(end_offset, previous_span_end)

                original_punctuation = text[previous_span_end:start_offset]
                original_span = text[start_offset:end_offset]
                corrected_span = self._output_formatter.format(
                    self._labels,
                    predicted_labels,
                    original_punctuation,
                    original_span,
                    original_labels)

                corrected_spans.append(corrected_span)
                previous_span_end = end_offset

            results.append(''.join(corrected_spans))

        return results

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path / 'punctuation_corrector.json'

    @classmethod
    def save_metadata(cls, class_name: str, labels: List[str], path: Path) -> None:
        metadata = {'class': class_name, 'labels': labels}
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
            config['labels'],
            output_formatter
        )

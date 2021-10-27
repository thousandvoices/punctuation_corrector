from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
from typing import List, Optional
import numpy as np

from ..common.preprocessing import TokenCase


class OutputFormatter(ABC):
    @staticmethod
    @abstractmethod
    def format(
            labels: List[str],
            predicted_scores: np.ndarray,
            predicted_case: Optional[np.ndarray],
            original_punctuation: str,
            original_span: str,
            original_scores: np.ndarray) -> str:

        pass


class Alignment(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    DASH = 3


MARK_ALIGNMENTS = defaultdict(lambda: Alignment.LEFT, {
    '—': Alignment.CENTER,
    '-': Alignment.DASH
})


class DefaultOutputFormatter(OutputFormatter):
    @staticmethod
    def format(
            labels: List[str],
            predicted_scores: np.ndarray,
            predicted_case: Optional[np.ndarray],
            original_punctuation: str,
            original_span: str,
            original_scores: np.ndarray) -> str:

        punctuation = original_punctuation
        if not all(original_scores == (predicted_scores > 0.5)):
            label_scores = zip(labels, predicted_scores)

            aligned_labels = defaultdict(list)
            label_scores = [(label, score) for label, score in label_scores if score > 0.5]
            sorted_scores = sorted(label_scores, key=lambda x: x[1], reverse=True)

            if len(sorted_scores) > 0:
                if MARK_ALIGNMENTS[sorted_scores[0][0]] == Alignment.DASH:
                    punctuation = sorted_scores[0][0]
                else:
                    for label, score in sorted_scores:
                        aligned_labels[MARK_ALIGNMENTS[label]].append(label)

                    punctuation = ''
                    for alignment in Alignment:
                        labels = aligned_labels[alignment]
                        if len(labels) > 0:
                            if alignment != Alignment.LEFT and len(punctuation) == 0:
                                punctuation += ' '
                            punctuation += labels[0]
                            if alignment != Alignment.RIGHT:
                                punctuation += ' '
            else:
                punctuation = ' '

        span = original_span
        if predicted_case is not None:
            predicted_case = np.argmax(predicted_case)
            if predicted_case == TokenCase.LOWER.value:
                span = span.lower()
            elif predicted_case == TokenCase.CAPITALIZE.value:
                span = span.capitalize()
            else:
                span = span.upper()

        return punctuation + span

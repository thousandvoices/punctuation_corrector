from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
from typing import List
import numpy as np


class OutputFormatter(ABC):
    @staticmethod
    @abstractmethod
    def format(
            labels: List[str],
            predicted_scores: np.ndarray,
            original_punctuation: str,
            original_span: str,
            original_scores: np.ndarray) -> str:

        pass


class Alignment(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


MARK_ALIGNMENTS = defaultdict(lambda: Alignment.LEFT, {
    '—': Alignment.CENTER
})


class DefaultOutputFormatter(OutputFormatter):
    @staticmethod
    def format(
            labels: List[str],
            predicted_scores: np.ndarray,
            original_punctuation: str,
            original_span: str,
            original_scores: np.ndarray) -> str:

        punctuation = original_punctuation
        if not all(original_scores == (predicted_scores > 0.5)):
            label_scores = zip(labels, predicted_scores)

            aligned_labels = defaultdict(list)
            for label, score in sorted(label_scores, key=lambda x: x[1], reverse=True):
                if score > 0.5:
                    aligned_labels[MARK_ALIGNMENTS[label]].append(label)

            punctuation = ''
            for alignment in Alignment:
                labels = aligned_labels[alignment]
                if len(labels) > 0:
                    if alignment != Alignment.LEFT:
                        punctuation += ' '
                    punctuation += labels[0]
                    if alignment != Alignment.RIGHT:
                        punctuation += ' '

            if len(punctuation) == 0:
                punctuation = ' '

        return punctuation + original_span

import numpy as np
from dataclasses import dataclass
from typing import List
from transformers import PreTrainedTokenizerBase


@dataclass
class EncodedText:
    tokens: np.ndarray
    labels: np.ndarray
    start_offsets: List[int]
    end_offsets: List[int]


def _clear_labels(label_ids: List[int]) -> np.ndarray:
    return np.zeros(len(label_ids), dtype=np.float32)


def encode_punctuation(
        tokenizer: PreTrainedTokenizerBase,
        label_ids: List[int],
        text: str,
        truncate_len: int) -> EncodedText:

    encoded_text = tokenizer(
        text.lower(),
        max_length=truncate_len,
        truncation=True,
        return_tensors='np',
        return_offsets_mapping=True)

    tokens = encoded_text['input_ids'][0]
    offsets = encoded_text['offset_mapping'][0]

    fixed_tokens = []
    fixed_offsets = []
    labels = []

    active_labels = _clear_labels(label_ids)

    for token, offset in zip(tokens, offsets):
        if token not in label_ids:
            fixed_tokens.append(token)
            fixed_offsets.append(offset)
            labels.append(active_labels)
            active_labels = _clear_labels(label_ids)
        else:
            active_labels[label_ids.index(token)] = 1

    start_offsets, end_offsets = zip(*fixed_offsets)
    return EncodedText(np.int64(fixed_tokens), np.float32(labels), start_offsets, end_offsets)

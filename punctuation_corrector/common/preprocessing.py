import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class EncodedText:
    tokens: np.ndarray
    labels: np.ndarray
    case_labels: np.ndarray
    start_offsets: List[int]
    end_offsets: List[int]


def _clear_labels(label_ids: List[int]) -> np.ndarray:
    return np.zeros(len(label_ids), dtype=np.float32)


class TokenCase(Enum):
    LOWER = 0
    CAPITALIZE = 1
    UPPER = 2


def _encode_case(token: str) -> TokenCase:
    if not token.isalpha():
        return TokenCase.LOWER

    if len(token) > 1 and token == token.upper():
        return TokenCase.UPPER
    elif token == token.capitalize():
        return TokenCase.CAPITALIZE
    else:
        return TokenCase.LOWER


def encode_punctuation(
        tokenizer: PreTrainedTokenizerBase,
        label_ids: List[int],
        text: str) -> EncodedText:

    encoded_text = tokenizer(
        text.lower(),
        truncation=False,
        return_tensors='np',
        return_offsets_mapping=True)

    tokens = encoded_text['input_ids'][0]
    offsets = encoded_text['offset_mapping'][0]

    fixed_tokens = []
    fixed_offsets = []
    labels = []
    case_labels = []

    active_labels = _clear_labels(label_ids)

    for token, offset in zip(tokens, offsets):
        if token not in label_ids:
            fixed_tokens.append(token)
            fixed_offsets.append(offset)
            labels.append(active_labels)
            active_labels = _clear_labels(label_ids)

            start, end = offset
            case_labels.append(_encode_case(text[start:end]).value)
        else:
            active_labels[label_ids.index(token)] = 1

    fixed_offsets[-1] = (len(text), len(text))
    start_offsets, end_offsets = zip(*fixed_offsets)
    return EncodedText(
        np.int64(fixed_tokens),
        np.float32(labels),
        np.int64(case_labels),
        start_offsets,
        end_offsets)


@dataclass
class EncodedOutput:
    labels: np.ndarray
    case_labels: Optional[np.ndarray]


def parse_output(output: np.ndarray, predict_case: bool) -> EncodedOutput:
    if predict_case:
        num_cases = len(TokenCase)
        return EncodedOutput(output[..., :-num_cases], output[..., -num_cases:])
    else:
        return EncodedOutput(output, None)

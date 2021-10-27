import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple

from .preprocessing import encode_punctuation


def _collate_tensors(tensors: List[np.ndarray]) -> np.ndarray:
    collated = np.zeros((
        len(tensors),
        max(len(tensor) for tensor in tensors),
        *(tensors[0].shape[1:])),
        dtype=tensors[0].dtype)

    for idx, tensor in enumerate(tensors):
        collated[idx, :len(tensor)] = np.array(tensor)

    return collated


def _collate_examples(
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    texts, labels, case_labels = list(zip(*batch))
    
    token_tensor = torch.from_numpy(_collate_tensors(texts))
    label_tensor = torch.from_numpy(_collate_tensors(labels))
    case_tensor = torch.from_numpy(_collate_tensors(case_labels))

    return token_tensor, label_tensor, case_tensor


class TextDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            texts: List[str],
            labels: List[str],
            shuffle: bool) -> None:

        self._shuffle = shuffle
        self._texts = texts
        self._tokenizer = tokenizer
        self._label_ids = self._tokenizer.convert_tokens_to_ids(labels)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoded_text = encode_punctuation(
            self._tokenizer, self._label_ids, self._texts[idx])

        return encoded_text.tokens, encoded_text.labels, encoded_text.case_labels

    def loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=_collate_examples,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=self._shuffle
        )

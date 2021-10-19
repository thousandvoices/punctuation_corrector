import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple

from .preprocessing import encode_punctuation


def _collate_examples(batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    texts, labels = list(zip(*batch))
    max_len = max(len(x) for x in texts)
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for idx, text in enumerate(texts):
        tokens[idx, :len(text)] = np.array(text)
    token_tensor = torch.from_numpy(tokens)

    label_tensor = np.zeros((len(batch), max_len, labels[0].shape[-1]), dtype=np.float32)
    for idx, label in enumerate(labels):
        label_tensor[idx, :len(label)] = np.array(label)
    label_tensor = torch.from_numpy(label_tensor)

    return token_tensor, label_tensor


class TextDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            texts: List[str],
            labels: List[str],
            truncate_len: int,
            shuffle: bool) -> None:

        self._truncate_len = truncate_len
        self._shuffle = shuffle
        self._texts = texts
        self._tokenizer = tokenizer
        self._label_ids = self._tokenizer.convert_tokens_to_ids(labels)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        encoded_text = encode_punctuation(
            self._tokenizer, self._label_ids, self._texts[idx], self._truncate_len)

        return encoded_text.tokens, encoded_text.labels

    def loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=_collate_examples,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=self._shuffle
        )

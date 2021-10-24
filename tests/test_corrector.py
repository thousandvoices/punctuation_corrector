import unittest
import numpy as np

from punctuation_corrector.inference import Corrector


class CorrectorTest(unittest.TestCase):
    def test_splits(self):
        self.assertEqual(Corrector._create_splits(100), [(0, 100, 0, 100)])
        self.assertEqual(Corrector._create_splits(511), [(0, 511, 0, 511)])
        self.assertEqual(Corrector._create_splits(512), [(0, 512, 0, 512)])

        self.assertEqual(Corrector._create_splits(513), [(0, 512, 0, 502), (492, 513, 10, 21)])
        self.assertEqual(Corrector._create_splits(520), [(0, 512, 0, 502), (492, 520, 10, 28)])

        self.assertEqual(
            Corrector._create_splits(1200),
            [(0, 512, 0, 502), (492, 1004, 10, 502), (984, 1200, 10, 216)])

    def test_splits_join(self):
        numbers = np.arange(2000, dtype=np.int32)
        splits = Corrector._create_splits(len(numbers))

        chunks = [
            numbers[input_start:input_end][result_start:result_end]
            for input_start, input_end, result_start, result_end
            in splits
        ]

        self.assertEqual(numbers.tolist(), np.concatenate(chunks, axis=0).tolist())

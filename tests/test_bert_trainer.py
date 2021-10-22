import unittest
from tempfile import TemporaryDirectory

from punctuation_corrector.training import BertTrainer
from punctuation_corrector.inference import Corrector


NUM_REPEATS = 500


class CorrectorsTest(unittest.TestCase):
    def _test_trainer(self, trainer, export_type):
        texts = ['comma, here', 'and nothing']
        trainer.fit(texts * NUM_REPEATS, eval_set=texts)

        with TemporaryDirectory() as temp_dir:
            trainer.save_corrector(temp_dir, export_type)
            corrector = Corrector.load(temp_dir)

            train_predictions = corrector.correct(texts)
            self.assertEqual(train_predictions, texts)

            predictions = corrector.correct([text.replace(',', ' ') for text in texts])
            self.assertEqual(predictions, texts)

            predictions = corrector.correct([text.replace(' no', ', no') for text in texts])
            self.assertEqual(predictions, texts)

    def test_pytorch(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            1,
            {'test': lambda y_true, y_pred: 0},
            labels=[',']
        )
        self._test_trainer(trainer, 'pytorch')

    def test_onnx(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            1,
            {},
            labels=[',']
        )
        self._test_trainer(trainer, 'onnx')

    def test_onnx_quantized(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            1,
            {},
            labels=[',']
        )
        self._test_trainer(trainer, 'onnx_quantized')

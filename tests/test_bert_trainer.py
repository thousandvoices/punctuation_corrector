import unittest
from tempfile import TemporaryDirectory

from punctuation_corrector.training import BertTrainer
from punctuation_corrector.inference import Corrector


NUM_REPEATS = 1000


class CorrectorsTest(unittest.TestCase):
    def _test_trainer(self, trainer, export_type):
        texts = ['comma, here, ', 'AND Nothing']
        trainer.fit(texts * NUM_REPEATS, eval_set=texts, num_epochs=1)

        with TemporaryDirectory() as temp_dir:
            trainer.save_corrector(temp_dir, export_type)
            corrector = Corrector.load(temp_dir)

            if trainer._predict_case:
                input_texts = [text.lower() for text in texts]
            else:
                input_texts = texts

            train_predictions = corrector.correct(input_texts)
            self.assertEqual(train_predictions, texts)

            predictions = corrector.correct([text.replace(',', '') for text in input_texts])
            self.assertEqual(predictions, texts)

            predictions = corrector.correct([text.replace(' no', ', no') for text in input_texts])
            self.assertEqual(predictions, texts)

    def test_pytorch(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            [','],
            True
        )
        self._test_trainer(trainer, 'pytorch')

    def test_onnx(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            [','],
            False
        )
        self._test_trainer(trainer, 'onnx')

    def test_onnx_quantized(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            [','],
            False
        )
        self._test_trainer(trainer, 'onnx_quantized')

    def test_saveload(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            [','],
            False
        )

        texts = ['comma here', 'and nothing']

        with TemporaryDirectory() as temp_dir:
            trainer.save_corrector(temp_dir, 'pytorch')
            corrector = Corrector.load(temp_dir)
            original = corrector.correct(texts)

            finetuner = BertTrainer.load(temp_dir)
            with TemporaryDirectory() as finetuned_dir:
                finetuner.save_corrector(finetuned_dir, 'pytorch')
                finetuned_corrector = Corrector.load(finetuned_dir)
                finetuned = finetuned_corrector.correct(texts)

        self.assertEqual(original, finetuned)

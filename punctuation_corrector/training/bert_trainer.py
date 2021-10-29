import os
import torch
import numpy as np
import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup
from transformers.convert_graph_to_onnx import convert, optimize, quantize
from sklearn.metrics import f1_score
import pytorch_lightning as pl

from ..common.preprocessing import TokenCase, parse_output
from ..common.text_dataset import TextDataset
from ..inference.corrector import Corrector
from ..inference.onnx_classifier import OnnxClassifier


TRUNCATE_LEN = 512


class LightningModel(pl.LightningModule):
    def __init__(
            self,
            base_model,
            train_loader,
            val_loader,
            predict_case,
            labels,
            lr,
            num_updates) -> None:
        super().__init__()

        self.base_model = base_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.predict_case = predict_case
        self.labels = labels
        self.num_updates = num_updates
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.case_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.default_device = 'cpu'
        self.lr = lr

    def to(self, device, *args, **kwargs):
        self.default_device = device
        return super().to(device, *args, **kwargs)

    def unpack_batch(self, batch):
        return [tensor.to(self.default_device)[:, :TRUNCATE_LEN] for tensor in batch]

    @staticmethod
    def collapse_sequence(x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, x.shape[-1])

    def forward(self, data):
        return self.base_model.forward(data, attention_mask=data > 0)[0]

    def training_step(self, batch, batch_idx):
        data, target, case = self.unpack_batch(batch)
        result = self.forward(data)

        result = parse_output(result, self.predict_case)
        loss = self.criterion.forward(result.labels, target)

        if result.case_labels is not None:
            loss += self.case_criterion.forward(
                result.case_labels.reshape(-1, len(TokenCase)), case.reshape(-1))

        self.log('loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data, target, case = self.unpack_batch(batch)

        result = self.forward(data)
        true_labels = target.detach().cpu().numpy().astype(np.int32)
        true_case = case.detach().cpu().numpy().astype(np.int32)
        predicted = result.detach().cpu().numpy()

        return {'labels': true_labels, 'case': true_case, 'predicted': predicted}

    def validation_epoch_end(self, outputs):
        labels = np.concatenate([self.collapse_sequence(x['labels']) for x in outputs], axis=0)
        case = np.concatenate([x['case'].flatten() for x in outputs], axis=0)
        predicted = np.concatenate(
            [self.collapse_sequence(x['predicted']) for x in outputs], axis=0)
        predicted = parse_output(predicted, self.predict_case)

        messages = []
        for idx, label in enumerate(self.labels):
            score = f1_score(labels[:, idx], predicted.labels[:, idx] > 0)
            messages.append(f'f1({label}): {score:.5f}')

        if self.predict_case:
            predicted_cases = np.argmax(predicted.case_labels, axis=-1)
            for case_class in TokenCase:
                score = f1_score(case == case_class.value, predicted_cases == case_class.value)
                messages.append(f'f1({case_class.name}): {score:.5f}')

        print(' '.join(messages))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, self.num_updates // 10, self.num_updates)

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'reduce_on_plateau': False,
            'frequency': 1
        }
        return [self.optimizer], [scheduler_config]


def _save_pytorch(
        path: Path,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase) -> None:

    tokenizer.save_pretrained(path)
    model.save_pretrained(path)


def _save_onnx(
        path: Path,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        quantize_model: bool) -> None:

    _save_pytorch(path, model, tokenizer)

    with TemporaryDirectory() as temp_dir:
        temp_model_path = Path(temp_dir) / 'temp.onnx'
        convert(
            framework='pt',
            model=str(path),
            output=temp_model_path,
            pipeline_name='ner',
            opset=11
        )
        optimized_path = optimize(temp_model_path)
        if quantize_model:
            optimized_path = quantize(optimized_path)
        target_path = OnnxClassifier.onnx_model_path(path)
        with open(optimized_path, 'rb') as src, gzip.open(target_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        os.remove(path / 'pytorch_model.bin')


class BertTrainer:
    EXPORT_FUNCTIONS = {
        'pytorch': _save_pytorch,
        'onnx': lambda path, model, tokenizer: _save_onnx(path, model, tokenizer, False),
        'onnx_quantized': lambda path, model, tokenizer: _save_onnx(path, model, tokenizer, True),
    }

    def __init__(
            self,
            model_path: str,
            num_layers: int,
            labels: List[str],
            predict_case: bool) -> None:

        self._model_path = model_path
        self._num_layers = num_layers
        self._labels = labels
        self._predict_case = predict_case

        num_labels = len(labels) if not predict_case else len(labels) + len(TokenCase)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            do_lower_case=False
        )
        self._model = AutoModelForTokenClassification.from_pretrained(
            self._model_path,
            config=AutoConfig.from_pretrained(
                self._model_path, num_hidden_layers=num_layers, num_labels=num_labels)
        )

    def fit(self, data: List[str], eval_set: List[str], num_epochs: int) -> None:
        grad_steps = 4

        if 'large' in self._model_path.split('-'):
            batch_size = 1
            lr = 5e-6
        else:
            batch_size = 4
            lr = 2e-5

        train_loader = TextDataset(
            self._tokenizer, data, self._labels, True
        ).loader(batch_size)
        val_loader = TextDataset(
            self._tokenizer, eval_set, self._labels, False
        ).loader(16)

        effective_batch_size = batch_size * grad_steps
        epoch_updates = (len(data) + effective_batch_size - 1) // effective_batch_size
        num_updates = num_epochs * epoch_updates

        model = LightningModel(
            self._model,
            train_loader,
            val_loader,
            self._predict_case,
            self._labels,
            lr,
            num_updates
        ).to('cuda')

        trainer = pl.Trainer(
            max_steps=num_updates,
            val_check_interval=min(20000, len(data) // batch_size),
            num_sanity_val_steps=0,
            accumulate_grad_batches=grad_steps,
            gpus=[0],
            checkpoint_callback=False,
            logger=False,
            weights_summary=None
        )
        trainer.fit(model)

    def save_corrector(self, path: str, export_type: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.EXPORT_FUNCTIONS[export_type](path, self._model, self._tokenizer)
        Corrector.save_metadata(
            export_type,
            self._labels,
            self._predict_case,
            self._num_layers,
            path
        )

    @staticmethod
    def load(path: str):
        config = Corrector.load_metadata(path)
        if config['class'] != 'pytorch':
            raise AssertionError('Only pytorch models can be loaded for finetuning')

        return BertTrainer(path, config['num_layers'], config['labels'], config['predict_case'])

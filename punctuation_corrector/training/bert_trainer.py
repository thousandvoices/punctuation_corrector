import os
import torch
import numpy as np
import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Mapping, Callable
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup
from transformers.convert_graph_to_onnx import convert, optimize, quantize
import pytorch_lightning as pl

from ..common.text_dataset import TextDataset
from ..inference.corrector import Corrector
from ..inference.onnx_classifier import OnnxClassifier


TRUNCATE_LEN = 512


class LightningModel(pl.LightningModule):
    def __init__(self, base_model, train_loader, val_loader, lr, num_updates, metrics) -> None:
        super().__init__()

        self.base_model = base_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_updates = num_updates
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.default_device = 'cpu'
        self.lr = lr
        self.metrics = metrics

    def to(self, device, *args, **kwargs):
        self.default_device = device
        return super().to(device, *args, **kwargs)

    def unpack_batch(self, batch):
        return [tensor.to(self.default_device) for tensor in batch]

    @staticmethod
    def collapse_sequence(x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, x.shape[-1])

    def forward(self, data):
        return self.base_model.forward(data, attention_mask=data > 0)[0]

    def training_step(self, batch, batch_idx):
        data, target = self.unpack_batch(batch)
        result = self.forward(data)
        loss = self.criterion.forward(result, target)
        self.log('loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data, target = self.unpack_batch(batch)

        result = self.forward(data)
        y_true = target.detach().cpu().numpy().astype(np.int32)
        y_pred = result.detach().cpu().numpy()

        return {'y_true': y_true, 'y_pred': y_pred}

    def validation_epoch_end(self, outputs):
        y_true = np.concatenate([self.collapse_sequence(x['y_true']) for x in outputs], axis=0)
        y_pred = np.concatenate([self.collapse_sequence(x['y_pred']) for x in outputs], axis=0)
        scores = {
            name: metric(y_true, y_pred) for name, metric in self.metrics.items()
        }
        message = ' '.join(f'{metric}: {score:.5f}' for metric, score in scores.items())
        print(message)

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


def _save_pytorch(path: Path, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)


def _save_onnx(path: Path, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
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
        quantized_path = quantize(optimized_path)
        target_path = OnnxClassifier.onnx_model_path(path)
        with open(quantized_path, 'rb') as src, gzip.open(target_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        os.remove(path / 'pytorch_model.bin')


class BertTrainer:
    EXPORT_FUNCTIONS = {
        'pytorch': _save_pytorch,
        'onnx': _save_onnx
    }

    def __init__(
            self,
            model_path: str,
            num_layers: int,
            num_epochs: int,
            metrics: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
            labels: List[str]) -> None:

        self.labels = labels
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.metrics = metrics

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            do_lower_case=False
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            config=AutoConfig.from_pretrained(
                self.model_path, num_hidden_layers=num_layers, num_labels=len(labels))
        )

    def fit(self, data: List[str], eval_set: Optional[List[str]] = None) -> None:
        grad_steps = 4

        if 'large' in self.model_path.split('-'):
            batch_size = 1
            lr = 5e-6
        else:
            batch_size = 4
            lr = 2e-5

        train_loader = TextDataset(
            self.tokenizer, data, self.labels, TRUNCATE_LEN, True
        ).loader(batch_size)
        if eval_set is not None:
            val_loader = TextDataset(
                self.tokenizer, eval_set, self.labels, TRUNCATE_LEN, False
            ).loader(16)
        else:
            val_loader = None

        effective_batch_size = batch_size * grad_steps
        epoch_updates = (len(data) + effective_batch_size - 1) // effective_batch_size
        num_updates = self.num_epochs * epoch_updates

        model = LightningModel(
            self.model,
            train_loader,
            val_loader,
            lr,
            num_updates,
            self.metrics
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

    def save_corrector(self, path: str, export_type: Optional[str] = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.EXPORT_FUNCTIONS[export_type](path, self.model, self.tokenizer)
        Corrector.save_metadata(
            export_type,
            self.labels,
            path
        )

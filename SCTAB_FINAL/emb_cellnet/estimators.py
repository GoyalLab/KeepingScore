from os.path import join
from typing import Dict, List

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.tuner.tuning import Tuner

import sys
sys.path.append("/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/scTab-devel")
from cellnet.datamodules import MerlinDataModule
from emb_cellnet.models import TabnetClassifier, LinearClassifier, MLPClassifier



class EstimatorCellTypeClassifier:

    datamodule: MerlinDataModule
    model: pl.LightningModule
    trainer: pl.Trainer

    def __init__(self, data_path: str, embedding: bool = False): # Edit
        self.data_path = data_path
        self.embedding = embedding

    def init_datamodule(
            self,
            batch_size: int = 2048,
            sub_sample_frac: float = 1.,
            dataloader_kwargs_train: Dict = None,
            dataloader_kwargs_inference: Dict = None,
            merlin_dataset_kwargs_train: Dict = None,
            merlin_dataset_kwargs_inference: Dict = None
    ):
        if self.embedding:
            class DummyDataModule: # placeholder
                def __init__(self):
                    self.batch_size = batch_size
                    self.train_dataset = self
                    self.val_dataset = self
                    self.partition_lens = [1]
                def train_dataloader(self): return None
                def val_dataloader(self): return None
                def test_dataloader(self): return None
                def predict_dataloader(self): return None

            self.datamodule = DummyDataModule()
        else:
            self.datamodule = MerlinDataModule(
                self.data_path,
                columns=['cell_type'],
                batch_size=batch_size,
                sub_sample_frac=sub_sample_frac,
                dataloader_kwargs_train=dataloader_kwargs_train,
                dataloader_kwargs_inference=dataloader_kwargs_inference,
                dataset_kwargs_train=merlin_dataset_kwargs_train,
                dataset_kwargs_inference=merlin_dataset_kwargs_inference
            )

    def init_model(self, model_type: str, model_kwargs):
        if model_type == 'tabnet':
            self.model = TabnetClassifier(**{**self.get_fixed_model_params(model_type), **model_kwargs})
            if self.embedding:
                model_kwargs.pop('train_set_size', None)
                model_kwargs.pop('val_set_size', None)
        elif model_type == 'linear':
            self.model = LinearClassifier(**{**self.get_fixed_model_params(model_type), **model_kwargs})
            if self.embedding:
                model_kwargs.pop('train_set_size', None)
                model_kwargs.pop('val_set_size', None)
        elif model_type == 'mlp':
            self.model = MLPClassifier(**{**self.get_fixed_model_params(model_type), **model_kwargs})
            if self.embedding:
                model_kwargs.pop('train_set_size', None)
                model_kwargs.pop('val_set_size', None)
        else:
            raise ValueError(f'model_type has to be in ["linear", "tabnet"]. You supplied: {model_type}')

    def init_trainer(self, trainer_kwargs):
        # Drop any accidental keys not supported by pl.Trainer
        # valid_keys = pl.Trainer.__init__.__code__.co_varnames
        # filtered_kwargs = {k: v for k, v in trainer_kwargs.items() if k in valid_keys}
        # self.trainer = pl.Trainer(**filtered_kwargs)
        self.trainer = pl.Trainer(**trainer_kwargs)

    def _check_is_initialized(self):
        if not self.model:
            raise RuntimeError('You need to call self.init_model before calling self.train')
        if not self.datamodule:
            raise RuntimeError('You need to call self.init_datamodule before calling self.train')
        if not self.trainer:
            raise RuntimeError('You need to call self.init_trainer before calling self.train')

    def get_fixed_model_params(self, model_type: str):
        if self.embedding:
            gene_dim = self.datamodule.train_dataset.embeddings.shape[1]
        else:
            gene_dim = len(pd.read_parquet(join(self.data_path, 'var.parquet')))
        model_params = {
            'gene_dim': gene_dim,  # <-- FIXED: use computed gene_dim
            'type_dim': len(pd.read_parquet(join(self.data_path, 'categorical_lookup/cell_type.parquet'))),
            'class_weights': np.load(join(self.data_path, 'class_weights.npy')),
            'child_matrix': np.load(join(self.data_path, 'cell_type_hierarchy/child_matrix.npy')),
            'train_set_size': sum(self.datamodule.train_dataset.partition_lens),
            'val_set_size': sum(self.datamodule.val_dataset.partition_lens),
            'batch_size': self.datamodule.batch_size,
        }
        if model_type in ['tabnet', 'mlp']:
            model_params['augmentations'] = np.load(join(self.data_path, 'augmentations.npy'))

        return model_params

    def find_lr(self, lr_find_kwargs, plot_results: bool = False):
        self._check_is_initialized()
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            **lr_find_kwargs
        )
        if plot_results:
            lr_finder.plot(suggest=True)

        return lr_finder.suggestion(), lr_finder.results

    def train(self, ckpt_path: str = None):
        self._check_is_initialized()
        self.trainer.fit(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=ckpt_path
        )

    def validate(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.validate(self.model, dataloaders=self.datamodule.val_dataloader(), ckpt_path=ckpt_path)

    def test(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(), ckpt_path=ckpt_path)

    def predict(self, dataloader=None, ckpt_path: str = None) -> np.ndarray:
        self._check_is_initialized()
        predictions_batched: List[torch.Tensor] = self.trainer.predict(
            self.model,
            dataloaders=dataloader if dataloader else self.datamodule.predict_dataloader(),
            ckpt_path=ckpt_path
        )
        return torch.vstack(predictions_batched).numpy()


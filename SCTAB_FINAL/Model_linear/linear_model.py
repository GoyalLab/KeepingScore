import os
import sys

import seaborn as sns
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping

torch.set_float32_matmul_precision('high')

from IPython import get_ipython

sys.path.append("/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL")
from emb_cellnet.estimators import EstimatorCellTypeClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch.utils.data import TensorDataset, DataLoader

print("[Info] Dataset loading...")
DATA_PATH = "/projects/b1042/GoyalLab/jaekj/merlin_cxg_2023_05_15_sf-log1p"

train_emb = torch.load("/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding/train_embedding.pt",  weights_only=True)
val_emb = torch.load("/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding/val_embedding.pt",  weights_only=True)
test_emb = torch.load("/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding/test_embedding.pt",  weights_only=True)


class EmbeddingDataset():
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels
        self.partition_lens = [len(embeddings)]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        if self.labels is not None:
            return {'X': x, 'cell_type': self.labels[idx]}
        return {'X': x}

class EmbeddingDataModule:
    def __init__(self, train_emb, val_emb, test_emb, batch_size=2048):
        self.train_dataset = EmbeddingDataset(train_emb['X'], train_emb['y_true'])
        self.val_dataset = EmbeddingDataset(val_emb['X'], val_emb['y_true'])
        self.test_dataset = EmbeddingDataset(test_emb['X'], test_emb['y_true'])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# config parameters
MODEL = 'cxg_2023_05_15_linear'
CHECKPOINT_PATH = os.path.join('/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/emb_tb_logs/linear/run1', MODEL)
LOGS_PATH = os.path.join('/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/emb_tb_logs/linear/run1', MODEL)

print("[Info] Load Linear model ...")
estim = EstimatorCellTypeClassifier(DATA_PATH, embedding=True)
SEED = 1
seed_everything(SEED)
estim.datamodule = EmbeddingDataModule(train_emb, val_emb, test_emb, batch_size=2048)

estim.init_trainer(
    trainer_kwargs={
        'max_epochs': 12,
        'default_root_dir': CHECKPOINT_PATH,
        'accelerator': 'gpu',
        'devices': 1,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'logger': [TensorBoardLogger(LOGS_PATH, name='default')],
        'log_every_n_steps': 100,
        'detect_anomaly': False,
        'enable_progress_bar': True,
        'enable_model_summary': False,
        'enable_checkpointing': True,
        'callbacks': [
            TQDMProgressBar(refresh_rate=50),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(filename='val_f1_macro_{epoch}_{val_f1_macro:.3f}', monitor='val_f1_macro', mode='max',
                            every_n_epochs=1, save_top_k=2),
            ModelCheckpoint(filename='val_loss_{epoch}_{val_loss:.3f}', monitor='val_loss', mode='min',
                            every_n_epochs=1, save_top_k=2),
            EarlyStopping(monitor='val_f1_macro', patience=20, mode='max', verbose=True)
        ],
    }
)
estim.init_model(
    model_type='linear',
    model_kwargs={
        'learning_rate': 0.0005,
        'weight_decay': 0.05,
        'optimizer': torch.optim.AdamW,
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_kwargs': {'step_size': 3, 'gamma': 0.9, 'verbose': True},
    },
)

print(ModelSummary(estim.model))

# Run LR finder
lr_find_res = estim.find_lr(
    lr_find_kwargs={
        'early_stop_threshold': 10.,
        'min_lr': 1e-8,
        'max_lr': 10.,
        'num_training': 120
    }
)

ax = sns.lineplot(x=lr_find_res[1]['lr'], y=lr_find_res[1]['loss'])
ax.set_xscale('log')
ax.set_ylim(2., top=9.)
ax.set_xlim(1e-6, 10.)
suggested_lr = lr_find_res[0]
print(f'Suggested learning rate: {suggested_lr}')

estim.init_trainer(
    trainer_kwargs={
        'max_epochs': 12,  # Reapply your intended value
        'default_root_dir': CHECKPOINT_PATH,
        'accelerator': 'gpu',
        'devices': 1,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'logger': [TensorBoardLogger(LOGS_PATH, name='default')],
        'log_every_n_steps': 100,
        'detect_anomaly': False,
        'enable_progress_bar': True,
        'enable_model_summary': False,
        'enable_checkpointing': True,
        'callbacks': [
            TQDMProgressBar(refresh_rate=50),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(filename='val_f1_macro_{epoch}_{val_f1_macro:.3f}', monitor='val_f1_macro', mode='max',
                            every_n_epochs=1, save_top_k=2),
            ModelCheckpoint(filename='val_loss_{epoch}_{val_loss:.3f}', monitor='val_loss', mode='min',
                            every_n_epochs=1, save_top_k=2),
            EarlyStopping(monitor='val_f1_macro', patience=20, mode='max', verbose=True)
        ],
    }
)

print("max_epochs (AFTER reset):", estim.trainer.max_epochs)

# Reinitialize model again (good practice if you want a clean start)
estim.init_model(
    model_type='linear',
    model_kwargs={
        'learning_rate': suggested_lr,
        'weight_decay': 0.05,
        'optimizer': torch.optim.AdamW,
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_kwargs': {'step_size': 3, 'gamma': 0.9, 'verbose': True},
    },
)

estim.train()
print("[Info] Training complete")

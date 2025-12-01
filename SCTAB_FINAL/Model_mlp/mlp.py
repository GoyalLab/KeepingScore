import os
import seaborn as sns
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping


torch.set_float32_matmul_precision('high')

# Import your estimator
import sys
sys.path.append("/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL")
from emb_cellnet.estimators import EstimatorCellTypeClassifier

from os.path import join

print("[Info] Dataset loading...")
DATA_PATH = '/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding'

train_emb = torch.load(join(DATA_PATH, "train_embedding.pt"), weights_only=True)
val_emb = torch.load(join(DATA_PATH, "val_embedding.pt"), weights_only=True)
test_emb = torch.load(join(DATA_PATH, "test_embedding.pt"), weights_only=True)

# Convert to numpy
X_train = train_emb["X"].cpu().numpy()  
y_train = train_emb["y_true"].squeeze().cpu().numpy().astype(int)

X_val = val_emb["X"].numpy()
y_val = val_emb["y_true"].squeeze().cpu().numpy().astype(int)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

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


### Init model
MODEL = "mlp"
CHECKPOINT_PATH = os.path.join('/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/emb_tb_logs/mlp/run1', MODEL)
LOGS_PATH = os.path.join('/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/emb_tb_logs/mlp/run1', MODEL)
DATA_PATH = "/projects/b1042/GoyalLab/jaekj/merlin_cxg_2023_05_15_sf-log1p"

print("[Info] Load MLP model ...")
estim = EstimatorCellTypeClassifier(DATA_PATH, embedding=True)  # no longer pass DATA_PATH, we are giving our own data
estim.datamodule = EmbeddingDataModule(train_emb, val_emb, test_emb)

seed_everything(1)

early_stop_callback = EarlyStopping(
    monitor="val_f1_macro",   # or "val_f1_macro" depending on what you care about
    patience=20,           # number of epochs with no improvement before stopping
    mode="max",            # "min" for loss, "max" for f1
    min_delta=1e-4,
)

estim.init_trainer(
    trainer_kwargs={
        'max_epochs': 48,
        'gradient_clip_val': 1.,
        'gradient_clip_algorithm': 'norm',
        'default_root_dir': CHECKPOINT_PATH,
        'accelerator': 'gpu',
        'devices': 1,
        'strategy': 'auto',
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'logger': [TensorBoardLogger(LOGS_PATH, name='default', version='version_7')],
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
            early_stop_callback,
        ],
    }
)

estim.init_model(
    model_type='mlp',
    model_kwargs={
        'learning_rate': 0.002,
        'weight_decay': 0.05,
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_kwargs': {
            'step_size': 1,
            'gamma': 0.9,
            'verbose': True
        },
        'optimizer': torch.optim.AdamW,
        'n_hidden': 8,
        'hidden_size': 128,
        'dropout': 0.1,
        'augment_training_data': False},
)

print(ModelSummary(estim.model))
# Find learning rate
lr_find_res = estim.find_lr(lr_find_kwargs={'early_stop_threshold': 10., 'min_lr': 1e-8, 'max_lr': 10., 'num_training': 100})

ax = sns.lineplot(x=lr_find_res[1]['lr'], y=lr_find_res[1]['loss'])
ax.set_xscale('log')
ax.set_ylim(4., top=7.)
ax.set_xlim(1e-6, 10.)

suggested_lr = lr_find_res[0]
print(f'Suggested learning rate: {suggested_lr}')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# After plotting
plt.savefig("lr_find_curve.png")
plt.close()


estim.init_trainer(
    trainer_kwargs={
        'max_epochs': 48,
        'gradient_clip_val': 1.,
        'gradient_clip_algorithm': 'norm',
        'default_root_dir': CHECKPOINT_PATH,
        'accelerator': 'gpu',
        'devices': 1,
        'strategy': 'auto',
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'logger': [TensorBoardLogger(LOGS_PATH, name='default', version='version_7')],
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
            early_stop_callback,
        ],
    }
)

estim.init_model(
    model_type='mlp',
    model_kwargs={
        'learning_rate':suggested_lr,
        'weight_decay': 0.05,
        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
        'lr_scheduler_kwargs': {
            'step_size': 1,
            'gamma': 0.9,
            'verbose': True
        },
        'optimizer': torch.optim.AdamW,
        'n_hidden': 8,
        'hidden_size': 128,
        'dropout': 0.1,
        'augment_training_data': False},
)

print("max_epochs (AFTER reset):", estim.trainer.max_epochs)

# Train using custom loaders
estim.train()
print("[Info] Training complete")


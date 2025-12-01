#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import scipy.sparse as sp
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

# ----------------------------
# Functions and Classes
# ----------------------------

def sigmoid_schedule(T, beta_min, beta_max, low=-6, high=6):
    t = torch.linspace(low, high, T)
    betas = torch.sigmoid(t) * (beta_max - beta_min) + beta_min
    return betas.clamp(max=0.999)

class ConditionalDenoiser(nn.Module):
    def __init__(self, T, num_classes, latent_dim, label_emb_dim, time_dim=64, hidden_dim=128):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        self.embed_time = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_emb_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.register_buffer("betas", torch.tensor([]))

    def forward(self, zt, t, y_label):
        device = next(self.parameters()).device
        zt = zt.to(device)
        t = t.to(device)
        y_label = y_label.to(device)
        t_embed = self.embed_time(t.float().unsqueeze(1) / float(self.T))
        y_label = y_label.squeeze(-1)  # [B]
        label_embed = self.label_emb(y_label)  # [B, label_emb_dim]
        cond = torch.cat([t_embed, label_embed], dim=1)
        return self.net(torch.cat([zt, cond], dim=1))

    def show_latent(self, x_0, t, noise):
        device = x_0.device
        betas = self.betas.to(device)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        t = t.to(device)
        sqrt_ab = torch.sqrt(alphas_bar[t].view(-1,1))
        sqrt_mab = torch.sqrt(1. - alphas_bar[t].view(-1,1))
        noise = noise.to(device)
        return sqrt_ab * x_0 + sqrt_mab * noise

class DiffusionModule(pl.LightningModule):
    def __init__(self, model, T, betas, lr, weight_decay, adam_betas):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer("betas", betas)
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas

    def forward(self, zt, t, y_label):
        return self.model(zt, t, y_label)

    def training_step(self, batch, batch_idx):
        x0, y_label = batch
        device = next(self.model.parameters()).device
        x0 = x0.to(device)
        y_label = y_label.to(device)
        noise = torch.randn_like(x0, device=device)
        t = torch.randint(0, self.T, (x0.shape[0],), device=device)
        xt = self.model.show_latent(x0, t, noise)
        pred_noise = self(xt, t, y_label)
        loss = F.mse_loss(pred_noise, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, y_label = batch
        device = next(self.model.parameters()).device
        x0 = x0.to(device)
        y_label = y_label.to(device)
        noise = torch.randn_like(x0, device=device)
        t = torch.randint(0, self.T, (x0.shape[0],), device=device)
        xt = self.model.show_latent(x0, t, noise)
        pred_noise = self(xt, t, y_label)
        val_loss = F.mse_loss(pred_noise, noise)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.adam_betas, weight_decay=self.weight_decay)

class UMAPCallback(Callback):
    def __init__(self, val_loader, every_n_epochs=50, save_dir="umap_plots", n_samples=2000):
        super().__init__()
        self.val_loader = val_loader
        self.every_n_epochs = every_n_epochs
        self.save_dir = save_dir
        self.n_samples = n_samples
        os.makedirs(save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0 or epoch == 0:
            self._plot_umap(pl_module, epoch)

    def _plot_umap(self, pl_module, epoch):
        pl_module.eval()
        xs, ys = [], []
        for xb, yb in self.val_loader:
            xb, yb = xb.to(pl_module.device), yb.to(pl_module.device)
            with torch.no_grad():
                t = torch.randint(0, pl_module.T, (xb.shape[0],), device=pl_module.device)
                noise = torch.randn_like(xb)
                xt = pl_module.model.show_latent(xb, t, noise)
                pred_noise = pl_module(xt, t, yb)
                betas = pl_module.betas.to(pl_module.device)
                alphas = 1. - betas
                alphas_bar = torch.cumprod(alphas, dim=0)
                sqrt_recip_alphas_bar = torch.sqrt(1. / alphas_bar[t]).view(-1,1)
                sqrt_recipm1_alphas_bar = torch.sqrt(1. / alphas_bar[t] - 1).view(-1,1)
                x0_pred = sqrt_recip_alphas_bar * xt - sqrt_recipm1_alphas_bar * pred_noise
            xs.append(x0_pred.cpu())
            ys.append(yb.cpu())
            if len(torch.cat(xs)) > self.n_samples:
                break
        xs = torch.cat(xs)[:self.n_samples]
        ys = torch.cat(ys)[:self.n_samples]
        zs = xs.numpy()
        labels = ys.numpy()
        reducer = umap.UMAP(random_state=42)
        emb = reducer.fit_transform(zs)
        plt.figure(figsize=(7,6))
        N = len(np.unique(labels))
        colors = plt.get_cmap("tab20")(np.linspace(0,1,N))
        scatter = plt.scatter(emb[:,0], emb[:,1], c=labels, cmap=ListedColormap(colors), s=2, alpha=0.7)
        plt.colorbar(scatter, label="Cell Types")
        plt.title(f"Denoised UMAP at Epoch {epoch}")
        plt.savefig(os.path.join(self.save_dir, f"Vanilla_diff_umap_epoch{epoch}.png"), dpi=300, bbox_inches="tight")
        plt.close()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model on embeddings")
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--emb_pred_path", type=str, required=True)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_min", type=float, default=1e-5)
    parser.add_argument("--beta_max", type=float, default=0.022)
    parser.add_argument("--low", type=float, default=-6)
    parser.add_argument("--high", type=float, default=6)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-9)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9,0.999))
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--denoising_umap_n_epochs", type=int, default=50)
    parser.add_argument("--umap_save_dir", type=str, default="umap_plots")
    parser.add_argument("--state_dir", type=str, default="./")
    parser.add_argument("--label_emb_dim", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--loglevel", type=str, default="INFO")
    args = parser.parse_args()

    # ----------------------------
    # Logging
    # ----------------------------
    logging.basicConfig(level=args.loglevel)
    logger = logging.getLogger(__name__)

    # ----------------------------
    # Multiprocessing safety
    # ----------------------------
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ----------------------------
    # Device
    # ----------------------------
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    device = torch.device("cuda" if gpu_available else "cpu")
    torch.set_float32_matmul_precision('high')
    logger.info(f"Using device: {device}, GPUs available: {gpu_count}")

    # ----------------------------
    # Load Data
    # ----------------------------
    logger.info("Loading AnnData embeddings...")

    train_emb = torch.load(os.path.join(args.emb_pred_path, "train_embedding.pt"), weights_only=True)
    val_emb = torch.load(os.path.join(args.emb_pred_path, "val_embedding.pt"), weights_only=True)
    
    X_train_emb = train_emb["X"].to(device)
    y_train = train_emb["y_true"].to(device)

    X_val_emb = val_emb["X"].to(device)
    y_val = val_emb["y_true"].to(device)

    uniques = np.unique(y_train.cpu().numpy())

    train_ds = TensorDataset(X_train_emb, y_train)
    val_ds   = TensorDataset(X_val_emb, y_val)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ----------------------------
    # Model & Module
    # ----------------------------
    betas_tensor = sigmoid_schedule(T=args.T, beta_min=args.beta_min, beta_max=args.beta_max,
                                    low=args.low, high=args.high)

    diff_model = ConditionalDenoiser(
        latent_dim=X_train_emb.shape[1],
        num_classes=164,
        T=args.T,
        label_emb_dim=args.label_emb_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim
    )
    diff_model.register_buffer("betas", betas_tensor)

    diffusion_module = DiffusionModule(
        model=diff_model,
        T=args.T,
        betas=betas_tensor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_betas=args.adam_betas
    )

    # ----------------------------
    # Callbacks
    # ----------------------------
    logger_tb = TensorBoardLogger("tb_logs", name="Vanilla Diffusion Model")
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=2)
    umap_cb = UMAPCallback(val_loader=val_loader, every_n_epochs=args.denoising_umap_n_epochs, save_dir=args.umap_save_dir)
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True)

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.n_devices,
        precision=32,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step"), early_stop_cb, umap_cb],
        logger=logger_tb,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True
    )

    # ----------------------------
    # Learning Rate Finder
    # ----------------------------
    torch.cuda.empty_cache()
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        diffusion_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        min_lr=1e-6,
        max_lr=1e-2,
        num_training=100
    )
    
    fig = lr_finder.plot(suggest=True)
    os.makedirs(args.state_dir,exist_ok=True)
    fig.savefig(os.path.join(args.state_dir, "lr_finder_plot_1.png"), dpi=300, bbox_inches="tight")
    new_lr = lr_finder.suggestion()
    if new_lr and new_lr < 1e-3:
        diffusion_module.lr = new_lr
        logger.info(f"Updated learning rate to {new_lr}")

    # ----------------------------
    # Training
    # ----------------------------
    logger.info("Starting training...")
    trainer.fit(diffusion_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training completed.")

    os.makedirs(args.state_dir, exist_ok=True)
    torch.save(diffusion_module.state_dict(), os.path.join(args.state_dir, "diffusion_module_state_1.pt"))
    logger.info(f"Model saved to {args.state_dir}")

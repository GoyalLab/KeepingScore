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
        device = self.device
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
        device = self.device
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
    """
    Visualize real-time denoising on the TRAIN SET.
    Shows:
        - original x0
        - noisy xt
        - denoised prediction x0_hat
    """
    def __init__(self, train_loader, every_n_epochs=50, save_dir="denoise_umap", n_samples=2000):
        super().__init__()
        self.train_loader = train_loader
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
        x0_list, xt_list, x0hat_list, labels_list = [], [], [], []

        with torch.no_grad():
            for xb, yb in self.train_loader:
                xb = xb.to(pl_module.device)
                yb = yb.to(pl_module.device)

                B = xb.shape[0]

                # sample random t for each item
                t = torch.randint(0, pl_module.T, (B,), device=pl_module.device)

                # add noise: xt = q(x0, t)
                noise = torch.randn_like(xb)
                xt = pl_module.model.show_latent(xb, t, noise)

                # predict noise
                pred_noise = pl_module(xt, t, yb)

                # reconstruct x0_hat = xt - predicted_noise (DDPM eps model)
                x0_hat = xt - pred_noise

                x0_list.append(xb.cpu())
                xt_list.append(xt.cpu())
                x0hat_list.append(x0_hat.cpu())
                labels_list.append(yb.cpu())

                if len(torch.cat(x0_list)) >= self.n_samples:
                    break

        # stack
        x0 = torch.cat(x0_list)[:self.n_samples]
        xt = torch.cat(xt_list)[:self.n_samples]
        x0hat = torch.cat(x0hat_list)[:self.n_samples]
        labels = torch.cat(labels_list)[:self.n_samples].numpy().squeeze()

        # combine for UMAP
        all_data = torch.cat([x0, xt, x0hat]).numpy()
        umap_labels = np.concatenate([
            labels,
            labels,
            labels
        ])

        stage = np.array(
            ["original"] * len(x0)
            + ["noisy"] * len(xt)
            + ["denoised"] * len(x0hat)
        )

        # UMAP
        reducer = umap.UMAP(random_state=42)
        emb = reducer.fit_transform(all_data)

        # Plot
        plt.figure(figsize=(10, 8))

        # original vs noisy vs denoised
        colors = {
            "original": "tab:blue",
            "noisy": "tab:orange",
            "denoised": "tab:green"
        }

        for st in ["original", "noisy", "denoised"]:
            mask = stage == st
            plt.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=3,
                alpha=0.5,
                c=colors[st],
                label=st,
            )

        plt.title(f"Denoising UMAP (Train Set) — Epoch {epoch}", fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"denoise_epoch{epoch}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(self.save_dir, f"denoise_epoch{epoch}.svg")
        )
        plt.close()

        pl_module.train()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model on embeddings")
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=4)
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
    torch.set_float32_matmul_precision('high')
    logger.info(f"Using GPUs: {gpu_count if gpu_available else 0}")

    # ----------------------------
    # Load Data (KEEP ON CPU!)
    # ----------------------------
    logger.info("Loading embeddings...")

    train_emb = torch.load(os.path.join(args.emb_pred_path, "train_embedding.pt"), 
                          weights_only=True)
    val_emb = torch.load(os.path.join(args.emb_pred_path, "val_embedding.pt"), 
                        weights_only=True)
    
    # Keep on CPU - DataLoader will transfer to GPU
    X_train_emb = train_emb["X"]
    y_train = train_emb["y_true"]
    X_val_emb = val_emb["X"]
    y_val = val_emb["y_true"]

    num_classes = len(torch.unique(y_train))
    logger.info(f"Training samples: {len(X_train_emb)}, Validation: {len(X_val_emb)}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Embedding dimension: {X_train_emb.shape[1]}")

    train_ds = TensorDataset(X_train_emb, y_train)
    val_ds = TensorDataset(X_val_emb, y_val)

    # DataLoaders with proper configuration
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.n_workers, 
        pin_memory=gpu_available,
        persistent_workers=args.n_workers > 0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.n_workers, 
        pin_memory=gpu_available,
        persistent_workers=args.n_workers > 0
    )

    # ----------------------------
    # Model & Module
    # ----------------------------
    betas_tensor = sigmoid_schedule(
        T=args.T, 
        beta_min=args.beta_min, 
        beta_max=args.beta_max,
        low=args.low, 
        high=args.high
    )

    diff_model = ConditionalDenoiser(
        latent_dim=X_train_emb.shape[1],
        num_classes=num_classes,
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
    logger_tb = TensorBoardLogger("tb_logs", name="conditional_diffusion")
    
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss", 
        mode="min", 
        save_top_k=2,
        filename="diffusion-{epoch:02d}-{val_loss:.4f}"
    )
    
    umap_cb = UMAPCallback(
        train_loader=train_loader,
        every_n_epochs=args.denoising_umap_n_epochs,
        save_dir=args.umap_save_dir,
    )
    
    early_stop_cb = EarlyStopping(
        monitor="val_loss", 
        patience=args.patience, 
        mode="min", 
        verbose=True
    )

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if gpu_available else "cpu",
        devices=args.n_devices if gpu_available else 1,
        precision=32,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step"), 
                  early_stop_cb, umap_cb],
        logger=logger_tb,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True
    )

    # ----------------------------
    # Learning Rate Finder (Optional)
    # ----------------------------
    if gpu_available:
        torch.cuda.empty_cache()
    
    logger.info("Running LR finder...")
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
    os.makedirs(args.state_dir, exist_ok=True)
    fig.savefig(os.path.join(args.state_dir, "lr_finder_plot_2.png"), 
               dpi=300, bbox_inches="tight")
    
    new_lr = lr_finder.suggestion()
    if new_lr and new_lr < args.lr:
        diffusion_module.lr = new_lr
        logger.info(f"Updated learning rate to {new_lr}")
    else:
        logger.info(f"Keeping original learning rate {args.lr}")

    # ----------------------------
    # Training
    # ----------------------------
    logger.info("Starting training...")
    trainer.fit(diffusion_module, train_dataloaders=train_loader, 
               val_dataloaders=val_loader)
    logger.info("Training completed.")

    # Save model
    os.makedirs(args.state_dir, exist_ok=True)
    save_path = os.path.join(args.state_dir, "diffusion_module_final.pt")
    torch.save(diffusion_module.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
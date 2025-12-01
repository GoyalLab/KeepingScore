#!/usr/bin/env python
# coding: utf-8

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap
from matplotlib.colors import ListedColormap

# =========================================================
# Utilities
# =========================================================

def sigmoid_schedule(T, beta_min, beta_max, low=-6, high=6):
    t = torch.linspace(low, high, T)
    betas = torch.sigmoid(t) * (beta_max - beta_min) + beta_min
    return betas.clamp(max=0.999)

# =========================================================
# Model (same as Lightning version)
# =========================================================

class ConditionalDenoiser(nn.Module):
    def __init__(self, T, num_classes, latent_dim, label_emb_dim=64, time_dim=64, hidden_dim=512):
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

        # diffusion schedule placeholder
        self.register_buffer("betas", torch.tensor([]))

    def forward(self, zt, t, y_label):
        t_embed = self.embed_time(t.float().unsqueeze(1) / float(self.T))
        label_embed = self.label_emb(y_label)
        cond = torch.cat([t_embed, label_embed], dim=1)
        return self.net(torch.cat([zt, cond], dim=1))

    def show_latent(self, x0, t, noise):
        betas = self.betas
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        sqrt_ab = torch.sqrt(alphas_bar[t].view(-1, 1))
        sqrt_mab = torch.sqrt(1 - alphas_bar[t].view(-1, 1))
        return sqrt_ab * x0 + sqrt_mab * noise

# =========================================================
# Manual Training Loop
# =========================================================

def train_epoch(model, loader, optimizer, device, T):
    model.train()
    total_loss = 0

    for x0, y in loader:
        x0 = x0.to(device)
        y = y.to(device)

        noise = torch.randn_like(x0)
        t = torch.randint(0, T, (x0.shape[0],), device=device)

        xt = model.show_latent(x0, t, noise)
        pred_noise = model(xt, t, y)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_epoch(model, loader, device, T):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x0, y in loader:
            x0 = x0.to(device)
            y = y.to(device)

            noise = torch.randn_like(x0)
            t = torch.randint(0, T, (x0.shape[0],), device=device)

            xt = model.show_latent(x0, t, noise)
            pred_noise = model(xt, t, y)
            loss = F.mse_loss(pred_noise, noise)

            total_loss += loss.item()

    return total_loss / len(loader)

# =========================================================
# Optional UMAP sampling
# =========================================================

def save_umap(model, loader, device, T, save_path):
    model.eval()

    xs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            t = torch.randint(0, T, (xb.shape[0],), device=device)
            noise = torch.randn_like(xb)

            xt = model.show_latent(xb, t, noise)
            pred_noise = model(xt, t, yb)

            betas = model.betas.to(device)
            alphas = 1 - betas
            alphas_bar = torch.cumprod(alphas, dim=0)

            sqrt_rec = torch.sqrt(1.0 / alphas_bar[t]).view(-1, 1)
            sqrt_rec_m1 = torch.sqrt(1.0 / alphas_bar[t] - 1).view(-1, 1)

            x0_pred = sqrt_rec * xt - sqrt_rec_m1 * pred_noise

            xs.append(x0_pred.cpu())
            ys.append(yb.cpu())

            if len(torch.cat(xs)) > 2000:
                break

    X = torch.cat(xs)[:2000].numpy()
    y = torch.cat(ys)[:2000].numpy()

    reducer = umap.UMAP(random_state=42)
    emb = reducer.fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(emb[:,0], emb[:,1], c=y, cmap="tab20", s=3)
    plt.colorbar()
    plt.title("Denoised UMAP")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_pred_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--T", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_emb = torch.load(os.path.join(args.emb_pred_path, "train_embedding.pt"), weights_only=True)
    val_emb   = torch.load(os.path.join(args.emb_pred_path, "val_embedding.pt"),   weights_only=True)

    X_train = train_emb["X"]
    y_train = train_emb["y_true"]
    X_val   = val_emb["X"]
    y_val   = val_emb["y_true"]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=args.batch)

    latent_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train.numpy()))

    betas = sigmoid_schedule(args.T, 1e-5, 0.022)

    model = ConditionalDenoiser(
        T=args.T,
        num_classes=n_classes,
        latent_dim=latent_dim,
        label_emb_dim=64,
        time_dim=64,
        hidden_dim=512,
    ).to(device)

    model.betas = betas.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = 1e9

    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.T)
        val_loss   = validate_epoch(model, val_loader, device, args.T)

        print(f"[Epoch {epoch}] Train={train_loss:.4f}  Val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "diffusion_model_wo_lightning.pt")
            print("  ✔ Saved best model")

        if epoch % 50 == 0:
            save_umap(model, val_loader, device, args.T, f"umap_epoch{epoch}.png")
            save_umap(model, val_loader, device, args.T, f"umap_epoch{epoch}.svg")

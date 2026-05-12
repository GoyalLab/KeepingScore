# %% [markdown]
# # Diffusion Model

# %% [markdown]
# ### Load modules & datasets

# %%
# === Train Conditional Diffusion Model on VAE Embeddings with Classifier-Based Guidance ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import pandas as pd
import anndata as ad
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
import numpy as np
import math

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### Load Data

# %%
batch_size = 256

os.chdir("/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/ExPert/anndata/splits")
print(f"Current working directory: {os.getcwd()}")

# Only load the train and validation sets for training
z_train_np = np.load('z_train.npy')
z_val_np = np.load('z_val.npy')

z_train = torch.tensor(z_train_np, dtype=torch.float32)
z_val = torch.tensor(z_val_np, dtype=torch.float32)

y_train_np = np.load('y_train.npy')
y_val_np = np.load('y_val.npy')

y_train = torch.tensor(y_train_np, dtype=torch.long)
y_val = torch.tensor(y_val_np, dtype=torch.long)

# Create data loaders
train_loader = DataLoader(TensorDataset(z_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(z_val, y_val), batch_size=batch_size, shuffle=False)

# %%
z_train.shape, z_val.shape

# %%
# check whether all train/val/test have all labels
print(len(y_train.unique()))
print(len(y_val.unique()))

# %%
os.chdir("/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/train_diffusion_model")
print(f"Current working directory: {os.getcwd()}")

os.makedirs("figures_updated", exist_ok=True)
# %% [markdown]
# #### Scheduling

# %%
def sigmoid_schedule(T, beta_min, beta_max, low: float = -6, high: float = 6):
    t = torch.linspace(low, high, T)
    betas = torch.sigmoid(t) * (beta_max - beta_min) + beta_min
    return betas.clamp(max=0.999)

# %%
T = 1000
beta_min=1e-5
beta_max=0.022
low = -6
high = 6

# %%
betas = sigmoid_schedule(T, beta_min=beta_min, beta_max=beta_max, low=low, high=high)

# %%
def plot_schedules(T, betas):
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas, axis=0)
    print("Final alpha value: ",alpha_bar[-1])

    # Plot beta
    plt.figure()
    plt.plot(np.arange(T), betas)
    plt.title(f"Sigmoid Beta Schedule (β_t over timesteps)")
    plt.xlabel("Timestep")
    plt.ylabel("β_t")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures_updated/schedule_sigmoid_beta.png")
    plt.close()

    # Plot alpha_bar
    plt.figure()
    plt.plot(np.arange(T), alpha_bar)
    plt.title("Cumulative ᾱ (product of (1 - β))")
    plt.xlabel("Timestep")
    plt.ylabel("ᾱ_t")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures_updated/schedule_cumulative_alpha.png")
    plt.show()

plot_schedules(T=T, betas=betas)

# %%
# ===== Model Configs =====
latent_dim = z_train.shape[1]
time_dim = 64
label_emb_dim = 64
epochs = 500
lr = 3e-4
min_lr = 1e-6
adam_betas = (0.9, 0.999)
wd_model = 1e-7
ema_decay = 0.9995
classifier_free_prob = 0.2

# learing rate scheduling 
warm_up_rate = 0.05

# early-stop parameters
patience = 30 # number of epochs to wait for improvement
patience_counter = 0

# %%
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: LongTensor [B] of timesteps in [0, T)
        returns: [B, dim] sinusoidal embedding
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # pad if dim is odd
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=t.device)], dim=1)
        return emb

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, cond_dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(feature_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, feature_dim)

    def forward(self, x, cond):
        B = x.shape[0]
        H = self.heads
        inner_dim = self.to_q.out_features
        d = inner_dim // H

        # compute queries, keys, values
        q = self.to_q(x).view(B, H, d)       # [B, H, d]
        k = self.to_k(cond).view(B, H, d)    # [B, H, d]
        v = self.to_v(cond).view(B, H, d)    # [B, H, d]

        # standard scaled dot-product attention across feature dimension
        attn_scores = torch.einsum('bhd,bhd->bh', q, k) / (d ** 0.5)  # or use self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, H, 1]

        # weighted sum of values per head
        out = (attn_weights * v).reshape(B, -1)  # [B, inner_dim]
        return self.to_out(out) + x

class FiLM(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        # initialize to near-identity: gamma around 1, beta around 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma_beta = self.proj(cond)  # [B, 2*feature_dim]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        # make scaling residual to keep initial behavior stable
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma * x + beta
    
    
class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        T,
        latent_dim,
        time_dim,
        n_classes,
        label_emb_dim=64,
        classifier_free_prob: float = 0.0,
    ):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.n_classes = n_classes
        self.label_emb_dim = label_emb_dim
        self.classifier_free_prob = classifier_free_prob

        # embeddings
        self.label_emb = nn.Embedding(n_classes + 1, label_emb_dim)
        self.embed_time = SinusoidalTimeEmbedding(time_dim)

        self.null_idx = n_classes
        self.cond_dim = time_dim + label_emb_dim

        # stage 1
        self.lin1 = nn.Linear(latent_dim + self.cond_dim, 512)
        self.fc1 = nn.Linear(512, 512)
        self.ln1 = nn.LayerNorm(512)
        self.attn1 = CrossAttention(512, self.cond_dim)
        self.film1 = FiLM(self.cond_dim, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.out1 = nn.Linear(512, latent_dim)

        # stage 2
        self.lin2 = nn.Linear(latent_dim * 2 + self.cond_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.ln2 = nn.LayerNorm(2048)
        self.attn2 = CrossAttention(2048, self.cond_dim)
        self.film2 = FiLM(self.cond_dim, 2048)
        self.dropout2 = nn.Dropout(0.1)
        self.out2 = nn.Linear(2048, latent_dim)

        self.betas = None

    def forward(self, zt, t, y_label):
        # classifier-free guidance dropout
        if self.training and self.classifier_free_prob > 0.0:
            keep_mask = torch.rand(y_label.shape[0], device=zt.device) > self.classifier_free_prob
            y_label = y_label.clone()
            y_label[~keep_mask] = self.null_idx

        label_embed = self.label_emb(y_label)          # [B, label_emb_dim]
        t_embed = self.embed_time(t)                   # [B, time_dim]
        cond = torch.cat([t_embed, label_embed], dim=1)

        # -----------------
        # Stage 1
        # -----------------
        x1 = torch.cat([zt, cond], dim=1)              # [B, latent_dim + cond_dim]
        h1 = self.lin1(x1)                             # [B, 512]
        h1 = self.fc1(h1)
        h1 = self.ln1(h1)
        h1 = self.attn1(h1, cond)
        h1 = self.film1(h1, cond)
        h1 = F.gelu(h1)
        h1 = self.dropout1(h1)

        h2 = self.fc1(h1)
        h2 = self.ln1(h2)
        h2 = self.attn1(h2, cond)
        h2 = self.film1(h2, cond)
        h2 = F.gelu(h2)

        pred1 = self.out1(h2)                          # [B, latent_dim]

        # -----------------
        # Stage 2
        # -----------------
        x2 = torch.cat([zt, pred1, cond], dim=1)      # [B, latent_dim*2 + cond_dim]
        g1 = self.lin2(x2)                             # [B, 2048]
        g1 = self.fc2(g1)
        g1 = self.ln2(g1)
        g1 = self.attn2(g1, cond)
        g1 = self.film2(g1, cond)
        g1 = F.gelu(g1)
        g1 = self.dropout2(g1)

        g2 = self.fc2(g1)
        g2 = self.ln2(g2)
        g2 = self.attn2(g2, cond)
        g2 = self.film2(g2, cond)
        g2 = F.gelu(g2)

        pred2 = self.out2(g2)                          # [B, latent_dim]

        return pred1 + pred2

    def show_latent(self, x_0, t, noise):
        if self.betas is None:
            raise ValueError("You must set self.betas before calling show_latent.")

        device = t.device
        betas = self.betas.to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        sqrt_ab = torch.sqrt(alphas_bar[t]).view(-1, 1)
        sqrt_mab = torch.sqrt(1.0 - alphas_bar[t]).view(-1, 1)
        return sqrt_ab * x_0 + sqrt_mab * noise
        

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]

# %%
### forward process visualization: model settings
model = ConditionalDenoiser(T=T, latent_dim=z_train.shape[1], n_classes=len(y_train.unique()), time_dim=time_dim, label_emb_dim=label_emb_dim).to(device)
model.betas = betas.to(device)
model.T = T

# %%
def compute_kl_qzt_prior(z0, t, alphas_bar, eps=1e-7):
    alpha_bar = alphas_bar[t].view(-1, 1).clamp(min=eps)      # [B,1]
    sqrt_alpha_bar = torch.sqrt(alpha_bar)                    # [B,1]
    var = (1.0 - alpha_bar).clamp(min=eps)                    # [B,1]
    log_var = torch.log(var)                                  # [B,1]
    mu = sqrt_alpha_bar * z0                                  # [B,D]

    kl_per_dim = mu.pow(2) + var - 1.0 - log_var             # [B,D]
    kl_per_sample = 0.5 * kl_per_dim.sum(dim=1)              # [B]
    return kl_per_sample

# %%
# UMAP parameters
timesteps_to_show = [0, 299, 499, 699, 999]
num_shows = len(timesteps_to_show)
n_cols = 5
n_rows = 1

# %%
## UMAP params
n_neighbors = 80
min_dist = 0.3
metric = "euclidean"

# %%
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0).to(device)

# %%
# Plot grid for UMAP visualization
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
axs = axs.flatten()

# ==== Put all tensors on the SAME DEVICE ====
subset_size = len(z_train)

z0_subset = z_train[:subset_size].to(device)
y_subset = y_train[:subset_size].to(device)

pert_numeric_labels_np = y_subset.detach().cpu().numpy()

kl_values = []

best_loss = float("inf")
best_state = None

# === 1. Fit UMAP ONCE on clean latent ===
z0_np = z0_subset.detach().cpu().numpy()

reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric=metric
)
reducer.fit(z0_np)   # 🔑 fit ONLY on t=0 data

for i, timestep in enumerate(timesteps_to_show):
    print(f"Generating UMAP and KL for timestep {timestep}")

    # Prepare timestep tensor on same device
    t_tensor = torch.full(
        (subset_size,), timestep,
        dtype=torch.long,
        device=device
    )

    # Generate noisy latent at timestep t
    if timestep == 0:
        z_t = z0_subset.clone()
    else:
        noise = torch.randn_like(z0_subset)
        z_t = model.show_latent(x_0=z0_subset, t=t_tensor, noise=noise)

    # Move to CPU only for UMAP
    z_t_np = z_t.detach().cpu().numpy()

    # no normalization
    # z_t_np = (z_t_np - z_t_np.mean(axis=0)) / (z_t_np.std(axis=0) + 1e-6)

    # Fit a fresh UMAP per timestep instead of transforming
    reducer_t = umap.UMAP(n_neighbors=n_neighbors, 
                        min_dist=min_dist, 
                        metric=metric,
                        random_state = 42)
    umap_emb = reducer_t.fit_transform(z_t_np)

    # Plot
    axs[i].scatter(
        umap_emb[:, 0], umap_emb[:, 1],
        c=pert_numeric_labels_np,
        s=5,
        cmap='tab20',
        alpha=0.8
    )
    axs[i].set_title(f"t = {timestep}")
    axs[i].axis('off')

    # KL
    kl_t = compute_kl_qzt_prior(z0_subset, t_tensor, alphas_bar)
    kl_values.append(kl_t.mean().item())

# Remove empty axes if any
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("./figures_updated/Forward_Diffusion_UMAP.png", dpi=300)
plt.savefig("./figures_updated/Forward_Diffusion_UMAP.svg")
plt.close()

# Plot KL divergence curve over timesteps
fig_kl, ax_kl = plt.subplots(figsize=(10, 5))
ax_kl.plot(timesteps_to_show, kl_values, marker='o')
ax_kl.set_xlabel("Timestep")
ax_kl.set_ylabel("Average KL Divergence")
ax_kl.set_title("KL Divergence per Timestep during Forward Diffusion")
ax_kl.grid(True)
plt.tight_layout()
plt.savefig("./figures_updated/L_Divergence_Over_Time.png", dpi=300)
plt.savefig("./figures_updated/KL_Divergence_Over_Time.svg")
plt.close()

print("FINAL KL divergence value:", kl_values[-1])

# %% [markdown]
# ## Training loop

# %% [markdown]
# ### Training the model

# %%
model_name = "Two_stage_FiLM_Diffusion"
schedule_name = "Sigmoid"

import os
os.makedirs("denoising_umap_updated", exist_ok=True)
# %%
# Inside your training loop:
best_losses = {}
best_states = {}
patience_counter = 0

print(f"\n=== Training with schedule: {schedule_name} ===")
alphas = 1.0 - betas
alphas_bar = alphas_bar.clone().detach().to(device)

def q_sample(z0, t):
    noise = torch.randn_like(z0)
    sqrt_ab = torch.sqrt(alphas_bar[t].view(-1, 1).clamp(min=1e-7))
    sqrt_one_minus_ab = torch.sqrt((1 - alphas_bar[t].view(-1, 1)).clamp(min=1e-7))
    return sqrt_ab * z0 + sqrt_one_minus_ab * noise

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=wd_model,
    betas=adam_betas
)

# Scheduler setup — warmup_steps and total_steps in steps, not epochs
total_steps = epochs * len(train_loader)
warmup_steps = int(warm_up_rate * total_steps)  # 5% warmup
scheduler = WarmupCosineLR(
    optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    min_lr=min_lr
)

# EMA setup
ema_state = {name: param.detach().clone() for name, param in model.named_parameters()}

best_loss = float("inf")
patience_counter = 0

train_loss_list = []
val_loss_list = []

global_step = 0  # track total steps for scheduler

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for z0, y in tqdm(train_loader, desc=f"{schedule_name} Epoch {epoch+1} [train]"):
        batch_size = z0.size(0)
        z0 = z0.to(device)
        y = y.to(device)
        t = torch.randint(0, T, (batch_size,), device=z0.device)

        # Forward noising
        zt = q_sample(z0, t)
        sqrt_alpha_bar = torch.sqrt(alphas_bar[t].view(-1, 1).clamp(min=1e-7))
        sqrt_one_minus_alpha_bar = torch.sqrt((1 - alphas_bar[t].view(-1, 1)).clamp(min=1e-7))
        true_noise = (zt - sqrt_alpha_bar * z0) / sqrt_one_minus_alpha_bar
        
        pred_noise = model(zt=zt, t=t, y_label=y)
        loss = F.mse_loss(pred_noise, true_noise, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update LR scheduler every step
        scheduler.step()
        global_step += 1

        # EMA update
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_state[name].mul_(ema_decay).add_(param.detach() * (1.0 - ema_decay))

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_train_loss = total_loss / total_samples
    train_loss_list.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    val_samples = 0

    # Snapshot current weights to restore after EMA evaluation
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}
    # Apply EMA for evaluation
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state:
                param.copy_(ema_state[name])

    with torch.no_grad():
        for z0_val, y_val_batch in tqdm(val_loader, desc=f"{schedule_name} Epoch {epoch+1} [val]"):
            batch_size_val = z0_val.size(0)
            z0_val = z0_val.to(device)
            y_val_batch = y_val_batch.to(device)
            t_val = torch.randint(0, T, (batch_size_val,), device=z0_val.device)

            zt_val = q_sample(z0_val, t_val)

            sqrt_alpha_bar_val = torch.sqrt(alphas_bar[t_val].view(-1, 1).clamp(min=1e-7))
            sqrt_one_minus_alpha_bar_val = torch.sqrt((1 - alphas_bar[t_val].view(-1, 1)).clamp(min=1e-7))
            true_noise_val = (zt_val - sqrt_alpha_bar_val * z0_val) / sqrt_one_minus_alpha_bar_val
            
            pred_noise_val = model(zt=zt_val, t=t_val, y_label=y_val_batch)
            loss_val = F.mse_loss(pred_noise_val, true_noise_val, reduction='mean')

            val_loss += loss_val.item() * batch_size_val
            val_samples += batch_size_val

    # Restore non-EMA weights to continue training
    model.load_state_dict(saved_state)

    avg_val_loss = val_loss / val_samples
    val_loss_list.append(avg_val_loss)

    print(
        f"{schedule_name} Epoch {epoch+1}: "
        f"Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}"
    )
    
    improvement_threshold = 1e-5
    if avg_val_loss < best_loss - improvement_threshold:
        best_loss = avg_val_loss
        best_state = model.state_dict()
        best_ema_state = {k: v.clone() for k, v in ema_state.items()}
        patience_counter = 0  

        # Build checkpoint dictionary
        checkpoint = {
            "model": model,                      # full model object (no need to recreate later)
            "model_state_dict": best_state,      # optional redundancy
            "ema_state_dict": best_ema_state,    # EMA weights
            "betas": betas,                      # your noise schedule
            "alphas_bar": alphas_bar.cpu().numpy(),
            "T": T,
            "schedule_name": schedule_name,
            "best_loss": best_loss,
            "optimizer_state_dict": optimizer.state_dict(),  # optional
            "train_loss_list": train_loss_list,
            "val_loss_list": val_loss_list,
        }

        torch.save(checkpoint, f"{model_name}_checkpoint_{schedule_name}_orig.pth")
        print(f"Saved checkpoint for '{schedule_name}' with val loss {best_loss:.6f}")
        
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s).")
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            # restore best weights before breaking
            model.load_state_dict(best_state)
            break
    
    # UMAP checkpointing every 30 epochs using EMA for visualization
    if (epoch + 1) % 30 == 0:
        
        # apply EMA weights
        saved_state = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_state:
                    param.copy_(ema_state[name])

        z0_all, y_all = [], []
        with torch.no_grad():
            for z0_batch, y_batch in train_loader:
                z0_batch = z0_batch.to(device)
                y_batch = y_batch.to(device)

                # start from pure noise (correct)
                x = torch.randn_like(z0_batch)

                # DDPM reverse loop
                for i in reversed(range(T)):
                    t_step = torch.full((x.size(0),), i, device=device)
                    pred_eps = model(x, t_step, y_batch)

                    beta = betas[i]
                    alpha = 1 - beta
                    alpha_bar = alphas_bar[i]

                    if i > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = 0

                    x = (1/torch.sqrt(alpha)) * (
                        x - (beta / torch.sqrt(1 - alpha_bar)) * pred_eps
                    ) + torch.sqrt(beta) * noise

                # x is now z0_hat
                z0_all.append(x.cpu())
                y_all.append(y_batch.cpu())

        # restore non-EMA weights
        model.load_state_dict(saved_state)

        z0_all = torch.cat(z0_all, dim=0).numpy()
        y_all = torch.cat(y_all, dim=0).numpy()

        z_umap = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        ).fit_transform(z0_all)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z_umap[:, 0], z_umap[:, 1], c=y_all, cmap='tab20', s=5, alpha=0.8)
        plt.colorbar(scatter)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.savefig(f"./denoising_umap_updated/{model_name}_model_UMAP_{schedule_name}_epoch{epoch+1}.png")
        plt.savefig(f"./denoising_umap_updated/{model_name}_model_UMAP_{schedule_name}_epoch{epoch+1}.svg")
        plt.close()

        # Save as best_z0_all only when checkpointing
        best_z0_all = z0_all
        best_y_all = y_all

# === LOSS PLOT ===
plt.figure(figsize=(8, 6))
epochs_range = range(1, len(train_loss_list) + 1)
plt.plot(epochs_range, train_loss_list, label="Train Loss", color="blue")
plt.plot(epochs_range, val_loss_list, label="Validation Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(f"{model_name} Loss Curve ({schedule_name})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./figures_updated/{model_name}_loss_curve_{schedule_name}.png")
plt.savefig(f"./figures_updated/{model_name}_loss_curve_{schedule_name}.svg")
plt.close()

# Summary
print("\n=== Best Losses by Schedule ===")
for name, loss in best_losses.items():
    print(f"{model_name} | {name}: {loss:.6f}")

# %% [markdown]
# 


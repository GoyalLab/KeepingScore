#!/usr/bin/env python
# coding: utf-8

# =========================
# Softmax Posterior Inference with Path Uncertainty
# =========================
import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

import sys
diffusion_directory = "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model"
sys.path.append(diffusion_directory)
import diffusion_model1  

# ----------------------------
# Argument Parser
# ----------------------------
parser = argparse.ArgumentParser(description="Softmax posterior inference with vanilla diffusion model")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default='diffusion_module_state.pt')
parser.add_argument("--n_paths", type=int, default=256, help="Number of Monte Carlo paths per observation")
parser.add_argument("--T_train", type=int, default=1000, help="Number of diffusion steps used in training")
parser.add_argument("--T_eval", type=int, default=1000, help="Number of diffusion steps for evaluation")
parser.add_argument("--beta_min", type=float, default=1e-5)
parser.add_argument("--beta_max", type=float, default=0.022)
parser.add_argument("--low", type=float, default=-6)
parser.add_argument("--high", type=float, default=6)
parser.add_argument("--save_dir", type=str, default="model_softmax/results/run_all")
parser.add_argument("--loglevel", type=str, default="INFO")
args = parser.parse_args()

# ----------------------------
# Logger
# ----------------------------
logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
logger = logging.getLogger(__name__)

# ----------------------------
# Device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ----------------------------
# Load embeddings
# ----------------------------
logger.info("Loading embeddings...")
train_emb = torch.load(join(args.data_path, "train_embedding.pt"), weights_only=True)
test_emb = torch.load(join(args.data_path, "test_embedding.pt"), weights_only=True)

X_train_emb = train_emb["X"].to(device)
y_train = train_emb["y_true"].to(device)
X_test_emb = test_emb["X"].to(device)
y_test = test_emb["y_true"].to(device)

# preserve original label order from training
unique_labels, inverse_idx = torch.unique(y_train, sorted=False, return_inverse=True)
label_to_idx = {int(lbl.item()): i for i, lbl in enumerate(unique_labels)}
idx_to_label = {i: int(lbl.item()) for i, lbl in enumerate(unique_labels)}

num_classes = len(unique_labels)
logger.info(f"Number of unique classes: {num_classes}, training samples: {X_train_emb.shape[0]}, test samples: {X_test_emb.shape[0]}")

# ----------------------------
# Load trained model
# ----------------------------
logger.info("Loading trained vanilla diffusion model...")
betas = diffusion_model1.sigmoid_schedule(
    T=args.T_train,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
    low=args.low,
    high=args.high
)

diff_model = diffusion_model1.ConditionalDenoiser(
    T=args.T_train,
    num_classes=num_classes,
    latent_dim=X_train_emb.shape[1],
    label_emb_dim=64,
    time_dim=64,
    hidden_dim=512
)
diff_model.register_buffer("betas", betas)

diffusion_module = diffusion_model1.DiffusionModule(
    model=diff_model,
    T=args.T_train,
    betas=betas,
    lr=3e-4,
    weight_decay=1e-9,
    adam_betas=(0.9, 0.999)
).to(device)

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint
missing = diffusion_module.load_state_dict(state_dict)
if missing.missing_keys or missing.unexpected_keys:
    logger.warning(
        "Loaded checkpoint with missing keys %s and unexpected keys %s",
        missing.missing_keys,
        missing.unexpected_keys,
    )
diffusion_module.eval()
if hasattr(diffusion_module, "freeze"):
    diffusion_module.freeze()

model = diffusion_module.model
model.to(device)
model.eval()
model.betas = betas.clone().to(device)
for p in model.parameters():
    p.requires_grad = False
logger.info("Model loaded successfully.")

def subsample_by_integrated_beta(betas_train, T_eval, device):
    """
    Subsample schedule by matching cumulative noise variance.
    """
    betas = betas_train.to(device)
    T_train = len(betas)
    
    # Cumulative variance σ²(t) = Σ β_i
    sigmasq = torch.cumsum(betas, dim=0)
    
    # Sample T_eval points uniformly in σ²-space
    sigmasq_eval = torch.linspace(0, sigmasq[-1], T_eval + 1, device=device)[1:]  # Skip 0
    
    # Find corresponding timesteps
    step_ids = torch.searchsorted(sigmasq, sigmasq_eval)
    step_ids = torch.clamp(step_ids, 0, T_train - 1)
    
    sigmasq_with_zero = torch.cat([torch.zeros(1, device=device), sigmasq])
    betas_sub = sigmasq_eval - sigmasq_with_zero[step_ids]
    
    return betas_sub, step_ids.cpu().numpy()


@torch.no_grad()
def softmax_posterior_latent(
    z0_obs, den_model, betas_train, T_train, T_eval, 
    beta_min, beta_max, low, high,
    n_paths=256, num_classes=10, device="cuda"):
    """
    Vectorized softmax posterior inference with proper beta subsampling.
    
    Args:
        z0_obs: [D] or [1, D] observation embedding
        den_model: trained denoising model
        betas_train: [T_train] training beta schedule
        T_train: Number of timesteps used during training
        T_eval: Number of timesteps to use for evaluation
        beta_min, beta_max, low, high: sigmoid schedule parameters
        n_paths: Number of Monte Carlo paths
        num_classes: Number of classes
        device: torch device
    
    Returns:
        map_label: int - MAP prediction
        posterior_mean: np.ndarray [num_classes] - mean posterior
        posterior_sem: np.ndarray [num_classes] - SEM across paths
        ll_per_path: np.ndarray [num_classes, n_paths] - raw log-likelihoods
    """
    # Shape setup
    if z0_obs.ndim == 1:
        z0_obs = z0_obs.unsqueeze(0)
    D = z0_obs.shape[-1]
    
    # Properly subsample betas using sigmoid schedule
    betas_sub, step_ids = subsample_by_integrated_beta(betas_train, T_eval, device)
    
    # Convert to alphas
    alphas = 1.0 - betas_sub
    sqrt_a = torch.sqrt(alphas)
    sqrt_1ma = torch.sqrt(1.0 - alphas)
    
    # Antithetic path sampling
    half = n_paths // 2
    eps = torch.randn(half, T_eval, D, device=device)
    eps_full = torch.cat([eps, -eps], dim=0)  # (n_paths, T_eval, D)
    
    # Forward diffuse shared paths
    z_path = torch.empty(n_paths, T_eval + 1, D, device=device)
    z_path[:, 0, :] = z0_obs
    
    for i in range(T_eval):
        z_path[:, i + 1, :] = sqrt_a[i] * z_path[:, i, :] + sqrt_1ma[i] * eps_full[:, i, :]
    
    # Expand across all classes
    z_path_expanded = z_path.repeat(num_classes, 1, 1)
    y_label = torch.arange(num_classes, device=device).repeat_interleave(n_paths)
    ll = torch.zeros(num_classes * n_paths, device=device)
    
    # Diffusion likelihood accumulation
    for i, t_idx in enumerate(step_ids):
        zt, zt1 = z_path_expanded[:, i, :], z_path_expanded[:, i + 1, :]
        t_vec = torch.full_like(y_label, t_idx, dtype=torch.long)
        score = den_model(zt1, t_vec, y_label)
        
        beta_t = betas_sub[i]  # Use subsampled beta directly
        
        # Likelihood terms
        ll += (score * (zt1 - zt)).sum(dim=1)
        ll += 0.5 * beta_t * (-(score**2).sum(dim=1) + (zt1 * score).sum(dim=1))
    
    # Reshape and compute posteriors
    ll_per_path = ll.view(num_classes, n_paths)
    posteriors_per_path = torch.softmax(ll_per_path, dim=0)
    
    posterior_mean = posteriors_per_path.mean(dim=1)
    posterior_sem = posteriors_per_path.std(dim=1) / np.sqrt(n_paths)
    
    map_label = torch.argmax(posterior_mean).item()
    
    return (
        map_label,
        posterior_mean.cpu().numpy(),
        posterior_sem.cpu().numpy(),
        ll_per_path.cpu().numpy()
    )

# ----------------------------
# Precompute class mean embeddings for TEST set
# ----------------------------
logger.info("Computing class mean embeddings for TEST set...")
unique_test_labels = torch.unique(y_test)
class_mean_embeds = []
class_n_cells = []

for true_label in unique_test_labels:
    class_idxs = (y_test == true_label).nonzero(as_tuple=True)[0]
    z_class = X_test_emb[class_idxs]
    mean_embed = z_class.mean(dim=0)
    class_mean_embeds.append(mean_embed)
    class_n_cells.append(len(class_idxs))

class_mean_embeds = torch.stack(class_mean_embeds).to(device)
logger.info(f"Computed {len(unique_test_labels)} class mean embeddings")

# ----------------------------
# Run inference on class means
# ----------------------------
logger.info("Running per-class softmax posterior inference on class means with path uncertainty...")
y_pred_all, y_true_all = [], []
posteriors_mean_all, posteriors_sem_all = [], []

save_root = os.path.join(args.save_dir, "class_results")
os.makedirs(save_root, exist_ok=True)

for class_idx, true_label_tensor in enumerate(unique_test_labels):
    true_label = int(true_label_tensor.item())
    n_cells = class_n_cells[class_idx]
    logger.info(f"[INFO] Running Softmax on test class {class_idx} (label={true_label}, n_cells={n_cells})...")

    # Get class mean embedding
    z0_obs_mean = class_mean_embeds[class_idx]

    # Run inference on the class mean
    map_label, post_mean, post_sem, ll_paths = softmax_posterior_latent(
        z0_obs=z0_obs_mean,
        den_model=model,
        betas_train=betas,
        T_train=args.T_train,
        T_eval=args.T_eval,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        low=args.low,
        high=args.high,
        n_paths=args.n_paths,
        num_classes=num_classes,
        device=device,
    )

    y_true_all.append(true_label)
    y_pred_all.append(map_label)
    posteriors_mean_all.append(post_mean)
    posteriors_sem_all.append(post_sem)

    logger.info(f"True label {true_label}: MAP = {map_label}")
    logger.info(f"Posterior: {post_mean}")

    # ----------------------------
    # Plot with path uncertainty error bars
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        range(len(post_mean)),
        post_mean,
        yerr=post_sem,                               # Path uncertainty
        capsize=3,                                    
        color="skyblue",
        edgecolor="black",
        ecolor="gray",                               
        alpha=0.9
    )

    ax.set_xlabel("Perturbation index", fontsize=12)
    ax.set_ylabel("Posterior probability", fontsize=12)
    ax.set_title(
        f"Posterior — True {true_label}, MAP {map_label} (n={n_cells})\n(error bars = path uncertainty)", 
        fontsize=14
    )

    # Annotate top-k classes
    top_k = 3
    sorted_idx = np.argsort(post_mean)[::-1][:top_k]
    for idx in sorted_idx:
        if post_mean[idx] > 1e-6:
            ax.annotate(
                f"{idx}\n{post_mean[idx]:.3f}±{post_sem[idx]:.3f}",   
                xy=(idx, post_mean[idx]),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5)
            )

    ax.set_ylim(0, max(post_mean + post_sem) * 1.25)  
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.7)
    fig.tight_layout()

    SAVE_PATH_CLASS = os.path.join(save_root, f"class_{class_idx}")
    os.makedirs(SAVE_PATH_CLASS, exist_ok=True)
    fig.savefig(f"{SAVE_PATH_CLASS}/steps_class_{class_idx}_posterior.png", dpi=150)
    fig.savefig(f"{SAVE_PATH_CLASS}/steps_class_{class_idx}_posterior.svg")
    plt.close(fig)

    # ----------------------------
    # Save results per class
    # ----------------------------
    np.save(os.path.join(SAVE_PATH_CLASS, "y_pred.npy"), map_label)
    np.save(os.path.join(SAVE_PATH_CLASS, "y_true.npy"), true_label)
    np.save(os.path.join(SAVE_PATH_CLASS, "posterior_mean.npy"), post_mean)
    np.save(os.path.join(SAVE_PATH_CLASS, "posterior_sem.npy"), post_sem)
    np.save(os.path.join(SAVE_PATH_CLASS, "ll_per_path.npy"), ll_paths)
    np.save(os.path.join(SAVE_PATH_CLASS, "n_cells.npy"), n_cells)

    logger.info(f"[INFO] Saved results for class {class_idx} (label={true_label})")

# ----------------------------
# Save aggregated results
# ----------------------------
np.save(os.path.join(args.save_dir, "y_pred_all.npy"), y_pred_all)
np.save(os.path.join(args.save_dir, "y_true_all.npy"), y_true_all)
np.save(os.path.join(args.save_dir, "posteriors_mean_all.npy"), posteriors_mean_all)
np.save(os.path.join(args.save_dir, "posteriors_sem_all.npy"), posteriors_sem_all)
logger.info(f"All {len(unique_test_labels)} class inferences completed successfully and saved.")
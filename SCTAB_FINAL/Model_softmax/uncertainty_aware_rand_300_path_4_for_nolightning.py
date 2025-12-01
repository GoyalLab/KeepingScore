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
parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
parser.add_argument("--beta_min", type=float, default=1e-5)
parser.add_argument("--beta_max", type=float, default=0.022)
parser.add_argument("--low", type=float, default=-6)
parser.add_argument("--high", type=float, default=6)
parser.add_argument("--save_dir", type=str, default="model_softmax/results/run_all")
parser.add_argument("--loglevel", type=str, default="INFO")
parser.add_argument("--sample_size", type=int, default=300, help="Number of random test samples to run inference on")
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
    T=args.T,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
    low=args.low,
    high=args.high
)

diff_model = diffusion_model1.ConditionalDenoiser(
    T=args.T,
    num_classes=num_classes,
    latent_dim=X_train_emb.shape[1],
    label_emb_dim=64,
    time_dim=64,
    hidden_dim=512
)
diff_model.register_buffer("betas", betas)

diffusion_module = diffusion_model1.DiffusionModule(
    model=diff_model,
    T=args.T,
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

# ----------------------------
# Softmax Posterior Function with Path Uncertainty
# ----------------------------
@torch.no_grad()
def softmax_posterior_latent(
    z0_obs,
    den_model,
    betas,
    T,
    n_paths=256,
    num_classes=10,
    device="cuda"
):
    """
    Vectorized softmax posterior inference with path-based uncertainty.
    
    Returns:
        map_label: int - MAP prediction
        posterior_mean: np.ndarray [num_classes] - mean posterior across paths
        posterior_sem: np.ndarray [num_classes] - SEM across paths (path uncertainty)
        ll_per_path: np.ndarray [num_classes, n_paths] - raw log-likelihoods per path
    """
    # --- shape setup ---
    if z0_obs.ndim == 1:
        z0_obs = z0_obs.unsqueeze(0)
    D = z0_obs.shape[-1]

    # --- diffusion constants ---
    alphas = 1. - betas
    sqrt_a = torch.sqrt(alphas)
    sqrt_1ma = torch.sqrt(1. - alphas)

    # --- antithetic path sampling ---
    half = n_paths // 2
    eps = torch.randn(half, T, D, device=device)
    eps_full = torch.cat([eps, -eps], dim=0)                     # (n_paths, T, D)

    # --- forward diffuse shared paths ---
    z_path = torch.empty(n_paths, T + 1, D, device=device)
    z_path[:, 0, :] = z0_obs
    for t in range(T):
        z_path[:, t + 1, :] = sqrt_a[t] * z_path[:, t, :] + sqrt_1ma[t] * eps_full[:, t, :]

    # --- expand across all classes ---
    z_path_expanded = z_path.repeat(num_classes, 1, 1)            # [num_classes*n_paths, T+1, D]
    y_label = torch.arange(num_classes, device=device).repeat_interleave(n_paths)
    ll = torch.zeros(num_classes * n_paths, device=device)

    # --- diffusion likelihood / energy accumulation ---
    for t in range(T):
        zt, zt1 = z_path_expanded[:, t, :], z_path_expanded[:, t + 1, :]
        t_vec = torch.full_like(y_label, t + 1, dtype=torch.long)

        score = den_model(zt1, t_vec, y_label)                   # [num_classes*n_paths, D]
        # main likelihood term
        ll += (score * (zt1 - zt)).sum(dim=1)
        ll += 0.5 * betas[t] * (-(score**2).sum(dim=1) + (zt1 * score).sum(dim=1))

    # --- reshape to [num_classes, n_paths] ---
    ll_per_path = ll.view(num_classes, n_paths)
    
    # --- compute posterior per path, then average ---
    posteriors_per_path = torch.softmax(ll_per_path, dim=0)  # [num_classes, n_paths]
    posterior_mean = posteriors_per_path.mean(dim=1)          # [num_classes]
    posterior_sem = posteriors_per_path.std(dim=1) / np.sqrt(n_paths)  # [num_classes]
    
    # --- MAP prediction ---
    map_label = torch.argmax(posterior_mean).item()
    
    return (
        map_label,
        posterior_mean.cpu().numpy(),
        posterior_sem.cpu().numpy(),
        ll_per_path.cpu().numpy()
    )


# ----------------------------
# Run inference
# ----------------------------
logger.info("Running per-class softmax posterior inference with path uncertainty...")
y_pred_all, y_true_all, posteriors_mean_all, posteriors_sem_all = [], [], [], []

save_root = os.path.join(args.save_dir, "class_results")
os.makedirs(save_root, exist_ok=True)

y_test_numpy = y_test.cpu().numpy()

for class_idx, true_label_tensor in enumerate(unique_labels):
    true_label = int(true_label_tensor.item())
    logger.info(f"[INFO] Running Softmax on test class {class_idx} (label={true_label})...")

    # ----------------------------
    # Collect all test samples for this label
    # ----------------------------
    class_idxs = np.where(y_test_numpy == true_label)[0]
    if len(class_idxs) == 0:
        logger.warning(f"No test samples found for label {true_label}")
        continue

    z0_obs_all = X_test_emb[class_idxs].to(device)

    # ----------------------------
    # Downsample if necessary
    # ----------------------------
    if len(z0_obs_all) > args.sample_size:
        torch.manual_seed(42)
        rand_idx = torch.randperm(len(z0_obs_all))[:args.sample_size]
        z0_obs_all = z0_obs_all[rand_idx]
        logger.info(f"[INFO] Downsampled from {len(class_idxs)} to {args.sample_size} samples.")
    else:
        logger.info(f"[INFO] Using {len(z0_obs_all)} samples for class {class_idx}.")

    # ----------------------------
    # Run inference for each sample in this class
    # ----------------------------
    y_pred, y_true = [], []
    posteriors_mean, posteriors_sem = [], []
    
    for i in range(z0_obs_all.shape[0]):
        logger.info(f"Processing sample {i+1}/{len(z0_obs_all)} for class {true_label}...")
        map_label, post_mean, post_sem, ll_paths = softmax_posterior_latent(
            z0_obs=z0_obs_all[i],
            den_model=model,
            betas=betas,
            T=args.T,
            n_paths=args.n_paths,
            num_classes=num_classes, 
            device=device,
        )

        y_true.append(true_label)
        y_pred.append(map_label)
        posteriors_mean.append(post_mean)
        posteriors_sem.append(post_sem)

    # ----------------------------
    # Class-level averaging
    # ----------------------------
    post_mean_stack = np.stack(posteriors_mean)      # [n_samples, num_classes]
    post_sem_stack = np.stack(posteriors_sem)        # [n_samples, num_classes]
    
    # Average across samples
    class_avg_posterior = post_mean_stack.mean(axis=0)     # [num_classes]
    class_avg_sem = post_sem_stack.mean(axis=0)            # [num_classes] - avg path uncertainty
    
    map_label_avg = int(np.argmax(class_avg_posterior))
    logger.info(f"True label {true_label}: MAP (avg over samples) = {map_label_avg}")

    # ----------------------------
    # Plot with path uncertainty error bars
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        range(len(class_avg_posterior)),
        class_avg_posterior,
        yerr=class_avg_sem,                          # Path uncertainty
        capsize=3,                                    
        color="skyblue",
        edgecolor="black",
        ecolor="gray",                               
        alpha=0.9
    )

    ax.set_xlabel("Perturbation index", fontsize=12)
    ax.set_ylabel("Posterior probability", fontsize=12)
    ax.set_title(
        f"Posterior — True {true_label}, MAP {map_label_avg}\n(error bars = path uncertainty)", 
        fontsize=14
    )

    # Annotate top-k classes
    top_k = 3
    sorted_idx = np.argsort(class_avg_posterior)[::-1][:top_k]
    for idx in sorted_idx:
        ax.annotate(
            f"{idx}\n{class_avg_posterior[idx]:.3f}±{class_avg_sem[idx]:.3f}",   
            xy=(idx, class_avg_posterior[idx]),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5)
        )

    ax.set_ylim(0, max(class_avg_posterior + class_avg_sem) * 1.25)  
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
    np.save(os.path.join(SAVE_PATH_CLASS, "y_pred.npy"), y_pred)
    np.save(os.path.join(SAVE_PATH_CLASS, "y_true.npy"), y_true)
    np.save(os.path.join(SAVE_PATH_CLASS, "posteriors_mean.npy"), posteriors_mean)
    np.save(os.path.join(SAVE_PATH_CLASS, "posteriors_sem.npy"), posteriors_sem)
    np.save(os.path.join(SAVE_PATH_CLASS, "class_avg_posterior.npy"), class_avg_posterior)
    np.save(os.path.join(SAVE_PATH_CLASS, "class_avg_sem.npy"), class_avg_sem)

    y_pred_all.extend(y_pred)
    y_true_all.extend(y_true)
    posteriors_mean_all.extend(posteriors_mean)
    posteriors_sem_all.extend(posteriors_sem)

    logger.info(f"[INFO] Saved results for class {class_idx} (label={true_label})")

# ----------------------------
# Save aggregated results
# ----------------------------
np.save(os.path.join(args.save_dir, "y_pred_all.npy"), y_pred_all)
np.save(os.path.join(args.save_dir, "y_true_all.npy"), y_true_all)
np.save(os.path.join(args.save_dir, "posteriors_mean_all.npy"), posteriors_mean_all)
np.save(os.path.join(args.save_dir, "posteriors_sem_all.npy"), posteriors_sem_all)
logger.info(f"All {num_classes} class inferences completed successfully and saved.")
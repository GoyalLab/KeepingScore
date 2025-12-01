#!/usr/bin/env python
# coding: utf-8

# =========================
# Softmax Posterior Inference using vanilla_model.py
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
import diffusion_model  

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
betas = diffusion_model.sigmoid_schedule(
    T=args.T,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
    low=args.low,
    high=args.high
)

diff_model = diffusion_model.ConditionalDenoiser(
    T=args.T,
    num_classes=num_classes,
    latent_dim=X_train_emb.shape[1],
    label_emb_dim=64,
    time_dim=64,
    hidden_dim=512
)
diff_model.register_buffer("betas", betas)

diffusion_module = diffusion_model.DiffusionModule(
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
# Softmax Posterior Function
# ----------------------------
@torch.no_grad()
def softmax_posterior_latent(
    z0_obs,
    den_model,
    betas,
    T,
    n_paths=256,
    num_classes=10,
    device="cuda",
    contrastive=True,
):
    """
    Vectorized softmax posterior inference (improved).
    Adds:
      • antithetic noise pairing  → lower Monte Carlo variance
      • optional contrastive energy term  → sharper class separation
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

        # --- optional contrastive term ---
        if contrastive:
            # subtract reference class (class 0) score at same zt1,t
            ref_score = score.view(num_classes, n_paths, D)[0]   # [n_paths, D]
            ref_score_exp = ref_score.repeat(num_classes, 1, 1).view(-1, D)
            contrast = (score - ref_score_exp).pow(2).sum(dim=1)
            # small negative weight → emphasize differences
            ll -= 0.25 * betas[t] * contrast

    # --- reshape, average over MC paths, normalize ---
    ll = ll.view(num_classes, n_paths)
    ll_all = ll.mean(dim=1)
    posterior = torch.softmax(ll_all, dim=0)
    map_label = torch.argmax(posterior).item()
    return map_label, posterior.cpu().numpy()


# ----------------------------
# Run inference
# ----------------------------
logger.info("Running per-class softmax posterior inference with 300 samples each...")
y_pred_all, y_true_all, posteriors_all = [], [], []

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
    y_pred, y_true, posteriors = [], [], []
    for i in range(z0_obs_all.shape[0]):
        logger.info(f"Processing sample {i+1}/{len(z0_obs_all)} for class {true_label}...")
        map_label, posterior = softmax_posterior_latent(
            z0_obs=z0_obs_all[i],
            den_model=model,
            betas=betas,
            T=args.T,
            n_paths=args.n_paths,
            num_classes=num_classes, 
            device=device,
        )

        # posterior is already a NumPy array
        post_np = posterior

        y_true.append(true_label)
        y_pred.append(map_label)
        posteriors.append(post_np)

    avg_post = np.mean(np.stack(posteriors), axis=0)
    map_label_avg = int(np.argmax(avg_post))
    logger.info(f"True label {true_label}: MAP (avg over samples) = {map_label_avg}")

    # ----------------------------
    # Plot the posterior histogram with error bars
    # ----------------------------
    post_stack = np.stack(posteriors)                
    post_mean = post_stack.mean(axis=0)
    post_std = post_stack.std(axis=0)                 

    map_label = int(np.argmax(post_mean))
    fig, ax = plt.subplots(figsize=(8, 5))

    # Add error bars using ax.bar with yerr
    bars = ax.bar(
        range(len(post_mean)),
        post_mean,
        yerr=post_std,                               
        capsize=3,                                    
        color="skyblue",
        edgecolor="black",
        ecolor="gray",                               
        alpha=0.9
    )

    ax.set_xlabel("Perturbation index", fontsize=12)
    ax.set_ylabel("Posterior probability", fontsize=12)
    ax.set_title(f"Posterior — True {true_label}, MAP {map_label}", fontsize=14)

    # Annotate top-k classes
    top_k = 3
    sorted_idx = np.argsort(post_mean)[::-1][:top_k]
    for idx in sorted_idx:
        ax.annotate(f"{idx}\n{post_mean[idx]:.3f}±{post_std[idx]:.3f}",   
                    xy=(idx, post_mean[idx]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))

    ax.set_ylim(0, max(post_mean + post_std) * 1.25)  
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    SAVE_PATH_CLASS = os.path.join(save_root, f"class_{class_idx}")
    os.makedirs(SAVE_PATH_CLASS, exist_ok=True)
    fig.savefig(f"{SAVE_PATH_CLASS}/steps_class_{class_idx}_posterior.png", dpi=150)
    plt.close(fig)

    # ----------------------------
    # Save results per class
    # ----------------------------
    np.save(os.path.join(SAVE_PATH_CLASS, "y_pred.npy"), y_pred)
    np.save(os.path.join(SAVE_PATH_CLASS, "y_true.npy"), y_true)
    np.save(os.path.join(SAVE_PATH_CLASS, "posteriors.npy"), posteriors)

    y_pred_all.extend(y_pred)
    y_true_all.extend(y_true)
    posteriors_all.extend(posteriors)

    logger.info(f"[INFO] Saved results for class {class_idx} (label={true_label})")

# ----------------------------
# Save aggregated results
# ----------------------------
np.save(os.path.join(args.save_dir, "y_pred_all.npy"), y_pred_all)
np.save(os.path.join(args.save_dir, "y_true_all.npy"), y_true_all)
np.save(os.path.join(args.save_dir, "posteriors_all.npy"), posteriors_all)
logger.info(f"All {num_classes} class inferences completed successfully and saved.")

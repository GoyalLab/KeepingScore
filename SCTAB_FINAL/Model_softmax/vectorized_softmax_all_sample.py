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
import vanilla_model  

torch.set_float32_matmul_precision("medium")
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
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
parser.add_argument("--class_chunk_size", type=int, default=10, help="Number of classes to process in one chunk")
parser.add_argument("--use_autocast", action="store_true", help="Use FP16 autocast for memory efficiency")
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
betas = vanilla_model.sigmoid_schedule(
    T=args.T,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
    low=args.low,
    high=args.high
)

diff_model = vanilla_model.ConditionalDenoiser(
    T=args.T,
    num_classes=num_classes,
    latent_dim=X_train_emb.shape[1],
    label_emb_dim=64,
    time_dim=64,
    hidden_dim=512
)
diff_model.register_buffer("betas", betas)

diffusion_module = vanilla_model.DiffusionModule(
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
    z0_batch, den_model, betas, T,
    n_paths=256, num_classes=10, device="cuda",
    eps_cache=None, class_chunk_size=10, use_autocast=True
):
    """
    Memory-efficient posterior inference:
    - Processes classes in chunks instead of expanding [B*num_classes, n_paths, T+1, D].
    - Optional FP16 autocast for lower VRAM usage.

    Returns:
        map_labels: [B]
        posteriors: [B, num_classes]
    """
    B, D = z0_batch.shape
    alphas = 1. - betas
    sqrt_a = torch.sqrt(alphas)
    sqrt_1ma = torch.sqrt(1. - alphas)

    # === Noise path reuse (antithetic or given) ===
    if eps_cache is None:
        eps = torch.randn(n_paths, T, D, device=device)
    else:
        eps = eps_cache.to(device)

    # === Precompute diffusion forward paths ===
    z_path = torch.empty(B, n_paths, T + 1, D, device=device)
    z_path[:, :, 0, :] = z0_batch[:, None, :]
    for t in range(T):
        z_path[:, :, t + 1, :] = sqrt_a[t] * z_path[:, :, t, :] + sqrt_1ma[t] * eps[None, :, t, :]

    # === Prepare log-likelihood storage ===
    ll_mean_all = torch.zeros(B, num_classes, device=device)

    # === Loop through class chunks to avoid [B*num_classes] explosion ===
    for c_start in range(0, num_classes, class_chunk_size):
        c_end = min(c_start + class_chunk_size, num_classes)
        this_chunk = c_end - c_start
        y_label = torch.arange(c_start, c_end, device=device)
        y_label = y_label.repeat_interleave(n_paths * B)

        ll_chunk = torch.zeros(B * this_chunk * n_paths, device=device)

        # === Main likelihood accumulation ===
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else torch.cuda.amp.autocast(enabled=False)
        with autocast_context:
            for t in range(T):
                zt = z_path[:, :, t, :].repeat(this_chunk, 1, 1).reshape(-1, D)
                zt1 = z_path[:, :, t + 1, :].repeat(this_chunk, 1, 1).reshape(-1, D)
                t_vec = torch.full_like(y_label, t + 1, dtype=torch.long)
                score = den_model(zt1, t_vec, y_label)
                ll_chunk += (score * (zt1 - zt)).sum(dim=1)
                ll_chunk += 0.5 * betas[t] * (-(score**2).sum(dim=1) + (zt1 * score).sum(dim=1))

        # === Reshape back and average across paths ===
        ll_chunk = ll_chunk.view(this_chunk, B, n_paths).permute(1, 0, 2)
        ll_mean_chunk = ll_chunk.mean(dim=2)
        ll_mean_all[:, c_start:c_end] = ll_mean_chunk

        torch.cuda.empty_cache()  # optional, helps prevent fragmentation

    # === Compute posterior and MAP ===
    posterior = torch.softmax(ll_mean_all, dim=1)
    map_labels = posterior.argmax(dim=1)

    return map_labels.cpu().numpy(), posterior.cpu().numpy()

# ----------------------------
# Run inference per class (vectorized)
# ----------------------------
logger.info("Running per-class softmax posterior inference with 300 samples each...")
y_pred_all, y_true_all, posteriors_all = [], [], []
save_root = os.path.join(args.save_dir, "class_results")
os.makedirs(save_root, exist_ok=True)
y_test_numpy = y_test.cpu().numpy()

for class_idx, true_label_tensor in enumerate(unique_labels):
    true_label = int(true_label_tensor.item())
    logger.info(f"[INFO] Running Softmax on test class {class_idx} (label={true_label})...")

    class_idxs = np.where(y_test_numpy == true_label)[0]
    if len(class_idxs) == 0:
        logger.warning(f"No test samples found for label {true_label}")
        continue

    z0_obs_all = X_test_emb[class_idxs].to(device)
    logger.info(f"[INFO] Using {len(z0_obs_all)} samples for class {class_idx}.")

    # ---- Cache noise paths once per class ----
    D = X_test_emb.shape[1]
    eps_half = torch.randn(args.n_paths // 2, args.T, D, device=device)
    eps_cache = torch.cat([eps_half, -eps_half], dim=0)  # antithetic
    logger.info(f"Cached noise paths: shape={eps_cache.shape}")

    # ---- Vectorized batch inference ----
    n_samples = z0_obs_all.shape[0]
    y_pred, y_true, posteriors = [], [], []

    for start in range(0, n_samples, args.batch_size):
        end = min(start + args.batch_size, n_samples)
        z_batch = z0_obs_all[start:end]
        logger.info(f"Processing batch {start+1}-{end}/{n_samples} for class {true_label}...")

        map_batch, post_batch = softmax_posterior_latent(
            z0_batch=z_batch,
            den_model=model,
            betas=betas,
            T=args.T,
            n_paths=args.n_paths,
            num_classes=num_classes,
            device=device,
            eps_cache=eps_cache,
            class_chunk_size=args.class_chunk_size, 
            use_autocast=args.use_autocast
        )
        y_true.extend([true_label] * len(map_batch))
        y_pred.extend(map_batch.tolist())
        posteriors.extend(post_batch)

    # ---- Aggregate per-class results ----
    post_stack = np.stack(posteriors)
    post_mean = post_stack.mean(axis=0)
    post_std = post_stack.std(axis=0)
    map_label = int(np.argmax(post_mean))
    logger.info(f"True label {true_label}: MAP (avg over samples) = {map_label}")

    # ---- Plot posterior ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(post_mean)), post_mean, yerr=post_std, capsize=3,
           color="skyblue", edgecolor="black", ecolor="gray", alpha=0.9)
    ax.set_xlabel("Perturbation index", fontsize=12)
    ax.set_ylabel("Posterior probability", fontsize=12)
    ax.set_title(f"Posterior — True {true_label}, MAP {map_label}", fontsize=14)
    top_k = 3
    sorted_idx = np.argsort(post_mean)[::-1][:top_k]
    for idx in sorted_idx:
        ax.annotate(f"{idx}\n{post_mean[idx]:.3f}±{post_std[idx]:.3f}",
                    xy=(idx, post_mean[idx]), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom',
                    fontsize=6, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))
    ax.set_ylim(0, max(post_mean + post_std) * 1.25)
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    SAVE_PATH_CLASS = os.path.join(save_root, f"class_{class_idx}")
    os.makedirs(SAVE_PATH_CLASS, exist_ok=True)
    fig.savefig(f"{SAVE_PATH_CLASS}/steps_class_{class_idx}_posterior.png", dpi=150)
    plt.close(fig)

    # ---- Save ----
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

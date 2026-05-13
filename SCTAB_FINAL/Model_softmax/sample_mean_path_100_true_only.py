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
diffusion_directory = "/projects/b1042/GoyalLab/jaekj/KeepingScore/SCTAB_FINAL/diffusion_model"
if diffusion_directory not in sys.path:
    sys.path.append(diffusion_directory)
import diffusion_model1

# ----------------------------
# Argument Parser
# ----------------------------
parser = argparse.ArgumentParser(description="Softmax posterior inference with vanilla diffusion model")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default='diffusion_module_state.pt')
parser.add_argument("--mean_path", type=str, required=True, help="Path to the precomputed class-mean embeddings (.npz with X_mean, y, and optional names).")
parser.add_argument("--n_paths", type=int, default=100, help="Number of Monte Carlo paths per observation")
parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
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


def load_class_mean_embeddings(mean_path, expected_latent_dim):
    logger.info("Loading class mean embeddings from %s", mean_path)
    mean_data = np.load(mean_path, allow_pickle=True)

    required_keys = {"X_mean", "y"}
    missing_keys = required_keys.difference(mean_data.files)
    if missing_keys:
        raise KeyError(f"Missing required arrays in {mean_path}: {sorted(missing_keys)}")

    x_mean = np.asarray(mean_data["X_mean"], dtype=np.float32)
    mean_labels = np.asarray(mean_data["y"])
    mean_names = np.asarray(mean_data["names"]) if "names" in mean_data.files else None

    if x_mean.ndim != 2:
        raise ValueError(f"X_mean must be 2D, got shape {x_mean.shape}")
    if mean_labels.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {mean_labels.shape}")
    if x_mean.shape[0] != mean_labels.shape[0]:
        raise ValueError(
            f"X_mean rows ({x_mean.shape[0]}) do not match y length ({mean_labels.shape[0]})"
        )
    if x_mean.shape[1] != expected_latent_dim:
        raise ValueError(
            f"X_mean latent dimension {x_mean.shape[1]} does not match expected {expected_latent_dim}"
        )
    if mean_names is not None and mean_names.shape[0] != mean_labels.shape[0]:
        raise ValueError(
            f"names length ({mean_names.shape[0]}) does not match y length ({mean_labels.shape[0]})"
        )

    ordered_labels = []
    ordered_indices = []
    for row_idx, label in enumerate(mean_labels.tolist()):
        label_int = int(label)
        if label_int not in label_to_idx:
            raise ValueError(f"Mean embedding label {label_int} is not present in the training labels")
        ordered_labels.append(label_int)
        ordered_indices.append(label_to_idx[label_int])

    logger.info("Loaded %d class mean embeddings", x_mean.shape[0])
    return (
        torch.from_numpy(x_mean).to(device),
        ordered_labels,
        ordered_indices,
        mean_names,
    )

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
def likelihood_true_class_only(
    z0_obs,
    true_label,
    den_model,
    betas,
    T,
    n_paths=4,
    device="cuda"
):
    if z0_obs.ndim == 1:
        z0_obs = z0_obs.unsqueeze(0)
    D = z0_obs.shape[-1]

    # diffusion constants
    alphas = 1. - betas
    alpha_bar_full = torch.cumprod(alphas, dim=0)
    sqrt_a = torch.sqrt(alphas)
    sqrt_1ma = torch.sqrt(1. - alphas)

    # --- antithetic sampling ---
    half = n_paths // 2
    eps = torch.randn(half, T, D, device=device)
    if n_paths % 2 == 0:
        eps_full = torch.cat([eps, -eps], dim=0)
    else:
        eps_full = torch.cat([eps, -eps, torch.randn(1, T, D, device=device)], dim=0)

    # --- forward diffusion ---
    z_path = torch.empty(n_paths, T + 1, D, device=device)
    z_path[:, 0, :] = z0_obs
    for t in range(T):
        z_path[:, t + 1, :] = sqrt_a[t] * z_path[:, t, :] + sqrt_1ma[t] * eps_full[:, t, :]

    # --- only TRUE CLASS ---
    y_label = torch.full((n_paths,), true_label, device=device, dtype=torch.long)

    ll = torch.zeros(n_paths, device=device)
    ll_traj = torch.zeros(n_paths, T, device=device)  # ← TRACK OVER TIME
    ll_comp = torch.zeros(n_paths, T, D, device=device)
    ll_step = torch.zeros(n_paths, T, device=device)

    for t in range(T):
        zt = z_path[:, t, :]
        zt1 = z_path[:, t + 1, :]
        t_vec = torch.full_like(y_label, t + 1)

        eps_hat = den_model(zt1, t_vec, y_label)
        a_bar = alpha_bar_full[t]
        sigma = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-12))
        score = -eps_hat / sigma
        dz = zt1 - zt

        ll_i_comp = -(score * dz)
        ll_i_comp += 0.5 * betas[t] * (-(score**2) - zt1 * score)

        increment = ll_i_comp.sum(dim=1)

        ll += increment
        ll_traj[:, t] = ll  # ← cumulative likelihood at each step
        ll_comp[:, t, :] = ll_i_comp
        ll_step[:, t] = increment

    return (
        ll.cpu().numpy(),          # final likelihood per path
        ll_traj.cpu().numpy(),     # [n_paths, T] ← convergence
        ll_comp.cpu().numpy(),      # [n_paths, T, D]
        ll_step.cpu().numpy()       # [n_paths, T] ← per-step increments
    )

# ----------------------------
# Run inference
# ----------------------------
logger.info("Running per-class softmax posterior inference with path uncertainty...")
y_pred_all, y_true_all, posteriors_mean_all, posteriors_sem_all = [], [], [], []
ll_paths_all_global, lci_all_global = [], []

save_root = os.path.join(args.save_dir, "class_results")

mean_embeddings, mean_labels, mean_label_indices, mean_names = load_class_mean_embeddings(
    args.mean_path,
    expected_latent_dim=X_train_emb.shape[1],
)

ll_final_all = []
ll_step_all = []
ll_traj_all = []
ll_comp_all = []

for class_idx, (true_label, true_label_idx) in enumerate(zip(mean_labels, mean_label_indices)):
    class_name = None if mean_names is None else str(mean_names[class_idx])
    logger.info(
        "[INFO] Running Softmax on class mean %d (label=%s%s)...",
        class_idx,
        true_label,
        "" if class_name is None else f", name={class_name}",
    )

    z0_obs_mean = mean_embeddings[class_idx]

    ll_final, ll_traj, ll_comp, ll_step = likelihood_true_class_only(
        z0_obs=z0_obs_mean,
        true_label=true_label_idx,
        den_model=model,
        betas=betas,
        T=args.T,
        n_paths=args.n_paths,
        device=device,
    )

    logger.info(f"Computed likelihood for true label {true_label}")

    # ----------------------------
    # Save results per class
    # ----------------------------
    ll_final_all.append(ll_final)
    ll_step_all.append(ll_step)
    ll_traj_all.append(ll_traj)
    ll_comp_all.append(ll_comp)
    logger.info(f"[INFO] Saved results for class {class_idx} (label={true_label})")


ll_final_all = np.stack(ll_final_all, axis=0)   # [C, P]
ll_step_all  = np.stack(ll_step_all, axis=0)    # [C, P, T]
ll_traj_all  = np.stack(ll_traj_all, axis=0)    # [C, P, T]
ll_comp_all  = np.stack(ll_comp_all, axis=0)    # [C, P, T, D]

os.makedirs(args.save_dir, exist_ok=True)
np.save(os.path.join(args.save_dir, "ll_final.npy"), ll_final_all)
np.save(os.path.join(args.save_dir, "ll_step.npy"), ll_step_all)
np.save(os.path.join(args.save_dir, "ll_traj.npy"), ll_traj_all)
np.save(os.path.join(args.save_dir, "ll_comp.npy"), ll_comp_all)

# ----------------------------
# Save aggregated results
# ----------------------------
logger.info(f"All {len(mean_labels)} class-mean inferences completed successfully and saved.")

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

save_root = "./Keeping_path256_long/"
os.makedirs(save_root, exist_ok=True)

# =========================================================
# Configuration
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "/projects/b1042/GoyalLab/jaekj/perturb-seq/Two_stage_FiLM_Diffusion_checkpoint_Sigmoid_orig.pth"

# =========================================================
# Load Model CLASS DEFINITIONS BEFORE LOADING CHECKPOINT
# =========================================================
import sys
sys.path.append("/projects/b1042/GoyalLab/jaekj/perturb-seq/")

from model import ConditionalDenoiser, FiLM, CrossAttention, SinusoidalTimeEmbedding

# register into safe unpickler globals
torch.serialization.add_safe_globals([
    ConditionalDenoiser,
    FiLM,
    CrossAttention,
    SinusoidalTimeEmbedding
])

# =========================================================
# Now it's safe to load checkpoint
# =========================================================
print("[INFO] Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Parameters
n_paths = 256  # Number of noise paths (use symmetric pairs)
T_eval = None  # Set to None to use full T from checkpoint

print(f"[INFO] Using device: {device}")

# =========================================================
# Load Model and Data
# =========================================================
print("[INFO] Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
T = checkpoint["T"]
T_eval = T_eval or T
betas = checkpoint["betas"].to(device)

# Recreate model (adjust these to match your architecture)
from model import ConditionalDenoiser, FiLM, CrossAttention, SinusoidalTimeEmbedding
torch.serialization.add_safe_globals([ConditionalDenoiser, FiLM, CrossAttention, SinusoidalTimeEmbedding])

latent_dim = 256
time_dim = 64
label_emb_dim = 64
n_classes = checkpoint["model"].n_classes

model = ConditionalDenoiser(
    T=T,
    latent_dim=latent_dim,
    n_classes=n_classes,
    time_dim=time_dim,
    label_emb_dim=label_emb_dim,
    classifier_free_prob=0.0
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"[INFO] Model loaded (T={T}, latent_dim={latent_dim})")

# Load test data
print("[INFO] Loading test embeddings...")
z_test = torch.tensor(
    np.load("/projects/b1042/GoyalLab/jaekj/perturb-seq/z_test.npy"),
    dtype=torch.float32
).to(device)

y_test = torch.tensor(
    np.load("/projects/b1042/GoyalLab/jaekj/perturb-seq/y_test.npy"),
    dtype=torch.long
).to(device)

# Map labels to contiguous indices
unique_labels = torch.unique(y_test).cpu().numpy()
label_map = {int(v): i for i, v in enumerate(unique_labels)}
num_classes = len(unique_labels)
y_test_mapped = torch.tensor([label_map[int(y)] for y in y_test.cpu().numpy()], device=device)

print(f"[INFO] Loaded {len(z_test)} samples with {num_classes} classes")

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_class_posterior(
    class_idx,
    true_label,
    avg_mean,
    avg_sem,
    num_classes,
    save_root,
    top_k=3,
    highlight_color="#2ecc71",
    other_color="skyblue"
):
    """
    Plot posterior probabilities with SEM error bars (path uncertainty)
    in the same style as the first code block.

    Parameters
    ----------
    class_idx : int
        Index of true class after mapping (0..C-1)
    true_label : int
        Original label value
    avg_mean : np.ndarray
        [num_classes] posterior mean for this class
    avg_sem : np.ndarray
        [num_classes] SEM (path uncertainty)
    num_classes : int
        Total number of classes
    save_root : str
        Directory where plots will be saved
    top_k : int
        Number of top classes to annotate
    highlight_color : str
        Color for true class
    other_color : str
        Color for all other classes
    """

    # -------------------------------
    # Create directory
    # -------------------------------
    os.makedirs(save_root, exist_ok=True)

    # -------------------------------
    # Colors: highlight true class
    # -------------------------------
    colors = [
        highlight_color if i == class_idx else other_color
        for i in range(num_classes)
    ]

    # -------------------------------
    # Begin plotting
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_classes)

    bars = ax.bar(
        x,
        avg_mean,
        yerr=avg_sem,
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        ecolor="gray",
        alpha=0.9
    )

    # -------------------------------
    # Annotate top-k classes
    # -------------------------------
    top_idx = np.argsort(avg_mean)[-top_k:][::-1]

    for idx in top_idx:
        ax.annotate(
            f"{idx}\n{avg_mean[idx]:.3f} ± {avg_sem[idx]:.3f}",
            xy=(idx, avg_mean[idx]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="black",
                lw=0.5
            ),
        )

    # -------------------------------
    # Axes / titles / formatting
    # -------------------------------
    ax.set_xlabel("Class index", fontsize=12)
    ax.set_ylabel("Posterior probability", fontsize=12)
    ax.set_title(
        f"Posterior — True {true_label} (mapped {class_idx})\nError bars = path uncertainty",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max(avg_mean + avg_sem) * 1.25)
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()

    # -------------------------------
    # Save (PNG + SVG, vector-safe)
    # -------------------------------
    out_png = os.path.join(save_root, f"class_{true_label}_posterior.png")
    out_svg = os.path.join(save_root, f"class_{true_label}_posterior.svg")

    fig.savefig(out_png, dpi=150)
    fig.savefig(out_svg, dpi=150)

    plt.close(fig)

    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_svg}")


# =========================================================
# Efficient Posterior Inference (from first code block)
# =========================================================
@torch.no_grad()
def diffusion_posterior_fast(
    z0_batch, model, betas, T_train, num_classes,
    n_paths=4, T_eval=None, device=None, class_chunk_size=50
):
    """
    Fully vectorized posterior inference:
    - Precomputes diffusion forward trajectories once
    - Reuses for all classes
    - Vectorizes model evaluations in class chunks
    """

    if device is None:
        device = z0_batch.device

    B, D = z0_batch.shape

    # -------------------------------
    # 1. Setup timestep subsampling
    # -------------------------------
    if T_eval is None or T_eval == T_train:
        step_ids = torch.arange(1, T_train + 1, device=device)   # [1..T]
        betas_sub = betas
    else:
        step_ids = torch.linspace(1, T_train, T_eval, dtype=torch.long, device=device)
        betas_sub = betas[step_ids - 1]

    alphas = 1. - betas_sub
    sqrt_a = torch.sqrt(alphas)
    sqrt_1ma = torch.sqrt(1. - alphas)

    # -------------------------------
    # 2. Generate symmetric noise
    # -------------------------------
    half = n_paths // 2
    eps_base = torch.randn(B, half, len(step_ids), D, device=device)
    eps = torch.cat([eps_base, -eps_base], dim=1)  # [B, n_paths, T_eval, D]

    # -------------------------------
    # 3. Precompute entire diffusion path
    #    z_path: [B, n_paths, T_eval+1, D]
    # -------------------------------
    T_use = len(step_ids)
    z_path = torch.empty(B, n_paths, T_use + 1, D, device=device)
    z_path[:, :, 0, :] = z0_batch[:, None, :]

    for i in range(T_use):
        t_idx = step_ids[i]
        noise = eps[:, :, i, :]
        z_prev = z_path[:, :, i, :]
        z_next = sqrt_a[i] * z_prev + sqrt_1ma[i] * noise
        z_path[:, :, i + 1, :] = z_next

    # -------------------------------
    # 4. Allocate outputs
    # -------------------------------
    ll_all = torch.zeros(B, num_classes, n_paths, device=device)

    # -------------------------------
    # 5. Process classes in chunks
    # -------------------------------
    for c_start in range(0, num_classes, class_chunk_size):
        c_end = min(c_start + class_chunk_size, num_classes)
        chunk_size = c_end - c_start

        # Shape: [chunk_size]
        y_chunk = torch.arange(c_start, c_end, device=device)

        # Repeat for all paths and batch items:
        # [chunk_size * B * n_paths]
        y_repeat = y_chunk[:, None, None].expand(chunk_size, B, n_paths)
        y_flat = y_repeat.reshape(-1)

        ll_chunk = torch.zeros(chunk_size, B, n_paths, device=device)

        # -------------------------------
        # 6. Loop through timesteps (vectorized across classes)
        # -------------------------------
        for i in range(T_use):
            t_global = step_ids[i]  # original t index
            beta_t = betas[t_global - 1]

            # z_t and z_{t+1}: shape [B, n_paths, D]
            zt = z_path[:, :, i, :]
            zt1 = z_path[:, :, i+1, :]

            # dz: [B, n_paths, D]
            dz = zt1 - zt

            # Expand to [chunk_size * B * n_paths, D]
            zt1_big = zt1[None, :, :, :].expand(chunk_size, B, n_paths, D)
            zt1_big = zt1_big.reshape(-1, D)

            dz_big = dz[None, :, :, :].expand(chunk_size, B, n_paths, D)
            dz_big = dz_big.reshape(-1, D)

            # timestep tensor
            t_vec = torch.full((chunk_size * B * n_paths,), t_global,
                               dtype=torch.long, device=device)

            # Model forward for this large chunk
            score = model(zt1_big, t_vec, y_flat)

            # Log-likelihood increment
            ll_i = (score * dz_big).sum(dim=1)
            ll_i += 0.5 * beta_t * (-(score**2).sum(dim=1)
                    + (zt1_big * score).sum(dim=1))

            # Accumulate into ll_chunk: reshape back
            ll_i = ll_i.view(chunk_size, B, n_paths)
            ll_chunk += ll_i

        # Save chunk to ll_all
        ll_all[:, c_start:c_end, :] = ll_chunk.permute(1, 0, 2)  # [B, C_chunk, n_paths]

    # -------------------------------
    # 7. Path-softmax → mean/SEM
    # -------------------------------
    probs = torch.softmax(ll_all, dim=1)  # softmax over classes (dim=1)
    mean_probs = probs.mean(2)            # [B, C]
    sem_probs = probs.std(2) / np.sqrt(n_paths)

    return mean_probs, sem_probs

# =========================================================
# Run Inference
# =========================================================
print(f"\n[INFO] Running posterior inference with {n_paths} paths...")
batch_size = 16
n_samples = len(z_test)

all_mean_probs = []
all_sem_probs = []
all_predictions = []

for start in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
    end = min(start + batch_size, n_samples)
    z0_batch = z_test[start:end]
    
    mean_probs, sem_probs = diffusion_posterior_fast(
        z0_batch=z0_batch,
        model=model,
        betas=betas,
        T_train=T,
        num_classes=num_classes,
        n_paths=n_paths,
        T_eval=T_eval,
        device=device
    )
    
    # Get predictions
    preds = mean_probs.argmax(dim=1)
    
    all_mean_probs.append(mean_probs.cpu().numpy())
    all_sem_probs.append(sem_probs.cpu().numpy())
    all_predictions.append(preds.cpu().numpy())

# Concatenate results
all_mean_probs = np.concatenate(all_mean_probs, axis=0)
all_sem_probs = np.concatenate(all_sem_probs, axis=0)
all_predictions = np.concatenate(all_predictions, axis=0)

# Map back to original labels
pred_labels_orig = [int(unique_labels[p]) for p in all_predictions]

# Calculate accuracy
y_test_np = y_test.cpu().numpy()
accuracy = (np.array(pred_labels_orig) == y_test_np).mean()
print(f"\n[RESULT] Overall Accuracy: {accuracy:.4f}")

# =========================================================
# Save Results
# =========================================================
print("\n[INFO] Saving results...")
np.save(os.path.join(save_root, "posterior_mean.npy"), all_mean_probs)
np.save(os.path.join(save_root, "posterior_sem.npy"), all_sem_probs)
np.save(os.path.join(save_root, "predictions.npy"), pred_labels_orig)

# Save per-sample results
results = []
for i in range(n_samples):
    results.append({
        "sample_index": i,
        "true_label": int(y_test_np[i]),
        "pred_label": int(pred_labels_orig[i]),
        "posterior_mean": all_mean_probs[i].tolist(),
        "posterior_sem": all_sem_probs[i].tolist()
    })

with open(os.path.join(save_root, "results.pkl"), "wb") as f:
    pickle.dump(results, f)

# =========================================================
# Generate Visualizations for Each Class
# =========================================================
print("\n[INFO] Generating class-level visualizations...")

for true_label in unique_labels:
    class_idx = label_map[true_label]
    class_mask = y_test_np == true_label
    
    if not class_mask.any():
        continue
    
    # Get posteriors for this class
    class_mean_probs = all_mean_probs[class_mask]
    class_sem_probs = all_sem_probs[class_mask]
    
    # Average across samples
    avg_mean = class_mean_probs.mean(axis=0)
    avg_sem = class_sem_probs.mean(axis=0)
    
    # Plot
    plot_class_posterior(
    class_idx=label_map[true_label],
    true_label=int(true_label),
    avg_mean=avg_mean,
    avg_sem=avg_sem,
    num_classes=num_classes,
    save_root=save_root
    )

print(f"\n[DONE] Results saved to {save_root}")
print(f"  - posterior_mean.npy: [N, {num_classes}]")
print(f"  - posterior_sem.npy: [N, {num_classes}]")
print(f"  - predictions.npy: [N]")
print(f"  - results.pkl: per-sample dictionary")
print(f"  - class_X_posterior.png: one plot per class")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
import math
import logging


# =========================================================
# Model Definitions
# =========================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
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
        batch_size = x.shape[0]
        heads = self.heads
        inner_dim = self.to_q.out_features
        dim_head = inner_dim // heads

        q = self.to_q(x).view(batch_size, heads, dim_head)
        k = self.to_k(cond).view(batch_size, heads, dim_head)
        v = self.to_v(cond).view(batch_size, heads, dim_head)

        attn_scores = torch.einsum("bhd,bhd->bh", q, k) / (dim_head ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)

        out = (attn_weights * v).reshape(batch_size, -1)
        return self.to_out(out) + x


class FiLM(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma_beta = self.proj(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
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
        classifier_free_prob=0.0,
    ):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.n_classes = n_classes
        self.label_emb_dim = label_emb_dim
        self.classifier_free_prob = classifier_free_prob

        self.label_emb = nn.Embedding(n_classes + 1, label_emb_dim)
        self.embed_time = SinusoidalTimeEmbedding(time_dim)

        self.null_idx = n_classes
        self.cond_dim = time_dim + label_emb_dim

        self.lin1 = nn.Linear(latent_dim + self.cond_dim, 512)
        self.fc1 = nn.Linear(512, 512)
        self.ln1 = nn.LayerNorm(512)
        self.attn1 = CrossAttention(512, self.cond_dim)
        self.film1 = FiLM(self.cond_dim, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.out1 = nn.Linear(512, latent_dim)

        self.lin2 = nn.Linear(latent_dim * 2 + self.cond_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.ln2 = nn.LayerNorm(2048)
        self.attn2 = CrossAttention(2048, self.cond_dim)
        self.film2 = FiLM(self.cond_dim, 2048)
        self.dropout2 = nn.Dropout(0.1)
        self.out2 = nn.Linear(2048, latent_dim)

        self.betas = None

    def forward(self, zt, t, y_label):
        if self.training and self.classifier_free_prob > 0.0:
            keep_mask = torch.rand(y_label.shape[0], device=zt.device) > self.classifier_free_prob
            y_label = y_label.clone()
            y_label[~keep_mask] = self.null_idx

        label_embed = self.label_emb(y_label)
        t_embed = self.embed_time(t)
        cond = torch.cat([t_embed, label_embed], dim=1)

        x1 = torch.cat([zt, cond], dim=1)
        h1 = self.lin1(x1)
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

        pred1 = self.out1(h2)

        x2 = torch.cat([zt, pred1, cond], dim=1)
        g1 = self.lin2(x2)
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

        pred2 = self.out2(g2)

        return pred1 + pred2


# =========================================================
# Argument Parser
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion posterior inference over perturbation mean embeddings")

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/model_keeping_score/results_path100",
                        help="Directory to save outputs")
    parser.add_argument("--n_paths", type=int, default=100,
                        help="Number of Monte Carlo paths per observation")
    parser.add_argument(
        "--mean_path",
        type=str,
        default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/datapoint_extraction/sample_mean/mean.npz",
        help="Path to perturbation-mean embeddings (.npz with X_mean, y, and optional names)",
    )
    parser.add_argument(
        "--z_test_path",
        type=str,
        default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/ExPert/anndata/splits/z_test.npy",
        help="Path to z_test.npy (used to derive label mapping)",
    )
    parser.add_argument(
        "--y_test_path",
        type=str,
        default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/ExPert/anndata/splits/y_test.npy",
        help="Path to y_test.npy (used to derive label mapping)",
    )
    parser.add_argument(
        "--perturbation_names_path",
        type=str,
        default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/ExPert/anndata/splits/perturbation_names.npy",
        help="Path to perturbation_names.npy for validating mean labels and names",
    )
    parser.add_argument("--loglevel", type=str, default="INFO")

    return parser.parse_args()


# =========================================================
# Logging
# =========================================================

args = parse_args()

logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
logger = logging.getLogger(__name__)

os.makedirs(args.save_dir, exist_ok=True)

# =========================================================
# Device
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# =========================================================
# Load Checkpoint
# =========================================================

# Remap modules so weights_only loading resolves serialized class names
ConditionalDenoiser.__module__ = "__main__"
FiLM.__module__ = "__main__"
CrossAttention.__module__ = "__main__"
SinusoidalTimeEmbedding.__module__ = "__main__"

torch.serialization.add_safe_globals([
    ConditionalDenoiser,
    FiLM,
    CrossAttention,
    SinusoidalTimeEmbedding,
    set,
    nn.Embedding,
    nn.Linear,
    nn.LayerNorm,
    nn.Dropout,
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    type(np.dtype(np.float32)),
])

logger.info("Loading checkpoint...")
checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
T = checkpoint["T"]
betas = checkpoint["betas"].to(device)

state_dict = checkpoint["model_state_dict"]
label_emb_dim = state_dict["label_emb.weight"].shape[1]
n_classes = state_dict["label_emb.weight"].shape[0] - 1
latent_dim = state_dict["out1.bias"].shape[0]
time_dim = state_dict["lin1.weight"].shape[1] - latent_dim - label_emb_dim

if time_dim <= 0:
    raise ValueError(
        f"Invalid checkpoint-derived dimensions: latent_dim={latent_dim}, "
        f"label_emb_dim={label_emb_dim}, time_dim={time_dim}"
    )

model = ConditionalDenoiser(
    T=T,
    latent_dim=latent_dim,
    n_classes=n_classes,
    time_dim=time_dim,
    label_emb_dim=label_emb_dim,
    classifier_free_prob=0.0,
).to(device)

model.load_state_dict(state_dict)
model.eval()
for p in model.parameters():
    p.requires_grad = False

logger.info(
    f"Model loaded (T={T}, latent_dim={latent_dim}, time_dim={time_dim}, "
    f"label_emb_dim={label_emb_dim}, n_classes={n_classes})"
)

# =========================================================
# Load Test Data (for label mapping only)
# =========================================================

logger.info("Loading test embeddings for label mapping...")
z_test = torch.tensor(np.load(args.z_test_path), dtype=torch.float32).to(device)
y_test = torch.tensor(np.load(args.y_test_path), dtype=torch.long).to(device)

if z_test.shape[1] != latent_dim:
    raise ValueError(
        f"Latent dimension mismatch: checkpoint expects {latent_dim}, "
        f"but z_test has shape {tuple(z_test.shape)}"
    )

# Build label mapping from training/test labels (preserve insertion order)
unique_labels = torch.unique(y_test).cpu().numpy()
label_to_idx = {int(v): i for i, v in enumerate(unique_labels)}
idx_to_label = {i: int(v) for i, v in enumerate(unique_labels)}
num_classes = len(unique_labels)

logger.info(f"Number of unique classes: {num_classes}, test samples: {len(z_test)}")

perturbation_names = np.load(args.perturbation_names_path, allow_pickle=True).astype(str)
if len(perturbation_names) != num_classes:
    raise ValueError(
        f"perturbation_names length ({len(perturbation_names)}) does not match "
        f"the number of unique y_test labels ({num_classes})"
    )

# =========================================================
# Load Perturbation Mean Embeddings
# =========================================================

def load_class_mean_embeddings(mean_path, expected_latent_dim):
    logger.info("Loading perturbation mean embeddings from %s", mean_path)
    mean_data = np.load(mean_path, allow_pickle=True)

    required_keys = {"X_mean", "y"}
    missing_keys = required_keys.difference(mean_data.files)
    if missing_keys:
        raise KeyError(f"Missing required arrays in {mean_path}: {sorted(missing_keys)}")

    x_mean = np.asarray(mean_data["X_mean"], dtype=np.float32)
    mean_labels_arr = np.asarray(mean_data["y"])
    mean_names = np.asarray(mean_data["names"]) if "names" in mean_data.files else None

    if x_mean.ndim != 2:
        raise ValueError(f"X_mean must be 2D, got shape {x_mean.shape}")
    if mean_labels_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {mean_labels_arr.shape}")
    if x_mean.shape[0] != mean_labels_arr.shape[0]:
        raise ValueError(
            f"X_mean rows ({x_mean.shape[0]}) do not match y length ({mean_labels_arr.shape[0]})"
        )
    if x_mean.shape[1] != expected_latent_dim:
        raise ValueError(
            f"X_mean latent dimension {x_mean.shape[1]} does not match expected {expected_latent_dim}"
        )
    if mean_names is not None and mean_names.shape[0] != mean_labels_arr.shape[0]:
        raise ValueError(
            f"names length ({mean_names.shape[0]}) does not match y length ({mean_labels_arr.shape[0]})"
        )

    ordered_labels = []
    ordered_indices = []
    for label in mean_labels_arr.tolist():
        label_int = int(label)
        if label_int not in label_to_idx:
            raise ValueError(
                f"Mean embedding label {label_int} is not present in the test labels"
            )
        if label_int >= len(perturbation_names):
            raise ValueError(
                f"Mean embedding label {label_int} is outside perturbation_names range "
                f"0..{len(perturbation_names) - 1}"
            )
        if mean_names is not None:
            expected_name = str(perturbation_names[label_int])
            observed_name = str(mean_names[len(ordered_labels)])
            if observed_name != expected_name:
                raise ValueError(
                    f"Mean embedding name mismatch for label {label_int}: "
                    f"mean.npz has {observed_name!r}, expected {expected_name!r}"
                )
        ordered_labels.append(label_int)
        ordered_indices.append(label_to_idx[label_int])

    logger.info("Loaded %d perturbation mean embeddings", x_mean.shape[0])
    return (
        torch.from_numpy(x_mean).to(device),
        ordered_labels,
        ordered_indices,
        mean_names,
    )


mean_embeddings, mean_labels, mean_label_indices, mean_names = load_class_mean_embeddings(
    args.mean_path,
    expected_latent_dim=latent_dim,
)

# =========================================================
# Likelihood Function (identical to reference script)
# =========================================================

@torch.no_grad()
def likelihood_true_class_only(
    z0_obs,
    true_label,
    den_model,
    betas,
    T,
    n_paths=4,
    device="cuda",
):
    if z0_obs.ndim == 1:
        z0_obs = z0_obs.unsqueeze(0)
    D = z0_obs.shape[-1]

    alphas = 1. - betas
    alpha_bar_full = torch.cumprod(alphas, dim=0)
    sqrt_a = torch.sqrt(alphas)
    sqrt_1ma = torch.sqrt(1. - alphas)

    eps_full = torch.randn(n_paths, T, D, device=device)

    z_path = torch.empty(n_paths, T + 1, D, device=device)
    z_path[:, 0, :] = z0_obs
    for t in range(T):
        z_path[:, t + 1, :] = sqrt_a[t] * z_path[:, t, :] + sqrt_1ma[t] * eps_full[:, t, :]

    y_label = torch.full((n_paths,), true_label, device=device, dtype=torch.long)

    ll      = torch.zeros(n_paths, device=device)
    ll_traj = torch.zeros(n_paths, T, device=device)
    ll_comp = torch.zeros(n_paths, D, device=device)
    ll_step = torch.zeros(n_paths, T, device=device)

    for t in range(T):
        zt  = z_path[:, t,     :]
        zt1 = z_path[:, t + 1, :]
        t_vec = torch.full_like(y_label, t + 1)

        eps_hat = den_model(zt1, t_vec, y_label)
        a_bar   = alpha_bar_full[t]
        sigma   = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-12))
        score   = -eps_hat / sigma
        dz      = zt1 - zt

        ll_i_comp  = -(score * dz)
        ll_i_comp += 0.5 * betas[t] * (-(score ** 2) - zt1 * score)

        increment = ll_i_comp.sum(dim=1)

        ll           += increment
        ll_traj[:, t] = ll
        ll_comp      += ll_i_comp
        ll_step[:, t] = increment

    return (
        ll.cpu().numpy(),       # [n_paths]        final likelihood per path
        ll_traj.cpu().numpy(),  # [n_paths, T]     cumulative likelihood trajectory
        ll_comp.cpu().numpy(),  # [n_paths, D]     per-dimension components
        ll_step.cpu().numpy(),  # [n_paths, T]     per-step increments
    )


# =========================================================
# Run Inference over Perturbation Mean Embeddings
# =========================================================

logger.info("Running per-perturbation likelihood inference...")

ll_final_all = []
ll_step_all  = []
ll_traj_all  = []
ll_comp_all  = []

for class_idx, (true_label, true_label_idx) in enumerate(zip(mean_labels, mean_label_indices)):
    class_name = None if mean_names is None else str(mean_names[class_idx])
    logger.info(
        "[INFO] Running inference on perturbation mean %d (label=%s%s)...",
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
        T=T,
        n_paths=args.n_paths,
        device=device,
    )

    ll_final_all.append(ll_final)
    ll_step_all.append(ll_step)
    ll_traj_all.append(ll_traj)
    ll_comp_all.append(ll_comp)

    logger.info(f"[INFO] Saved results for perturbation {class_idx} (label={true_label})")

# =========================================================
# Stack and Save
# =========================================================

ll_final_all = np.stack(ll_final_all, axis=0)  # [C, P]
ll_step_all  = np.stack(ll_step_all,  axis=0)  # [C, P, T]
ll_traj_all  = np.stack(ll_traj_all,  axis=0)  # [C, P, T]
ll_comp_all  = np.stack(ll_comp_all,  axis=0)  # [C, P, D]

os.makedirs(args.save_dir, exist_ok=True)
np.save(os.path.join(args.save_dir, "ll_final.npy"), ll_final_all)
np.save(os.path.join(args.save_dir, "ll_step.npy"),  ll_step_all)
np.save(os.path.join(args.save_dir, "ll_traj.npy"),  ll_traj_all)
np.save(os.path.join(args.save_dir, "ll_comp.npy"),  ll_comp_all)

logger.info(
    f"All {len(mean_labels)} perturbation-mean inferences completed successfully and saved to {args.save_dir}."
)

import anndata as ad
import pandas as pd
import scanpy as sc
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load embedding
# -------------------------------
test = torch.load("test_embedding.pt", map_location="cpu")

X = test["X"]
y = test["y"]

# convert to numpy
if isinstance(X, torch.Tensor):
    X = X.numpy()
if isinstance(y, torch.Tensor):
    y = y.numpy().reshape(-1)

adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame({"y": y.astype(str)})
)

# -------------------------------
# Replace with shortened names
# -------------------------------
import yaml
with open('/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/scTab-devel/notebooks/data_augmentation/shortend_cell_types.yaml', 'r') as f:
    shortend_cell_types = yaml.safe_load(f)

adata.obs["y"] = adata.obs["y"].replace(shortend_cell_types)

# -------------------------------
# Compute t-SNE
# -------------------------------
sc.pp.neighbors(adata, use_rep="X")
sc.tl.tsne(adata)

# -------------------------------
# Plot + Save PNG/SVG
# -------------------------------
plt.rcParams['figure.figsize'] = (5,5)

ax = sc.pl.tsne(
    adata,
    color="y",
    legend_fontsize="x-small",
    ncols=1,
    title="t-SNE of latent space",
    show=False
)

# Convert Axes → Figure
fig = ax.get_figure()

fig.savefig("tsne_embedding.png", dpi=300, bbox_inches="tight")
fig.savefig("tsne_embedding.svg", bbox_inches="tight")

print("[INFO] Saved tsne_embedding.{png,svg}")

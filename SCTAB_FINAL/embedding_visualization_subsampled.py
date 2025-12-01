import anndata as ad
import pandas as pd
import scanpy as sc
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import seaborn as sns
from sklearn.manifold import TSNE
import yaml


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# Paths
DATA_PATH = '/projects/b1042/GoyalLab/jaekj/merlin_cxg_2023_05_15_sf-log1p'
YAML_PATH = '/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/scTab-devel/notebooks/data_augmentation/shortend_cell_types.yaml'

# Load embeddings
print("Loading embeddings...")
test = torch.load("test_embedding.pt", map_location="cpu")

# Load test labels
print("Loading test labels...")
from os.path import join
y_test_names = dd.read_parquet(join(DATA_PATH, 'test'), columns=['cell_type']).compute()['cell_type'].values

# Load cell type mapping
print("Loading cell type mapping...")
from os.path import join
cell_type_mapping = pd.read_parquet(join(DATA_PATH, 'categorical_lookup/cell_type.parquet'))

# Load shortened cell type names
with open(YAML_PATH, 'r') as f:
    shortend_cell_types = yaml.safe_load(f)

# Create reverse mapping from index to cell type name
idx_to_celltype = dict(zip(cell_type_mapping.index, cell_type_mapping['label']))

# Map test labels to cell type names
y_test_celltype_names = [idx_to_celltype[idx] for idx in y_test_names]

# Apply shortened names
y_test_display_names = [shortend_cell_types.get(name, name) for name in y_test_celltype_names]

# Convert embeddings to numpy if needed
if isinstance(test, torch.Tensor):
    test_embeddings = test.numpy()
else:
    test_embeddings = test

print(f"Embedding shape: {test_embeddings.shape}")
print(f"Number of labels: {len(y_test_display_names)}")

# Subsample if dataset is too large (optional, for faster computation)
MAX_SAMPLES = 50000
if test_embeddings.shape[0] > MAX_SAMPLES:
    print(f"Subsampling to {MAX_SAMPLES} cells for faster t-SNE computation...")
    indices = np.random.choice(test_embeddings.shape[0], MAX_SAMPLES, replace=False)
    test_embeddings_subset = test_embeddings[indices]
    y_test_subset = [y_test_display_names[i] for i in indices]
else:
    test_embeddings_subset = test_embeddings
    y_test_subset = y_test_display_names

# Compute t-SNE
print("Computing t-SNE (this may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
embeddings_2d = tsne.fit_transform(test_embeddings_subset)

# Create DataFrame for plotting
tsne_df = pd.DataFrame({
    'tsne1': embeddings_2d[:, 0],
    'tsne2': embeddings_2d[:, 1],
    'cell_type': y_test_subset
})

# Get unique cell types and assign colors
unique_cell_types = sorted(set(y_test_subset))
n_types = len(unique_cell_types)
print(f"Number of unique cell types: {n_types}")

# Create color palette
if n_types <= 20:
    palette = sns.color_palette("tab20", n_types)
else:
    palette = sns.color_palette("husl", n_types)

# Create the plot
fig, ax = plt.subplots(figsize=(16, 12))

for i, cell_type in enumerate(unique_cell_types):
    mask = tsne_df['cell_type'] == cell_type
    ax.scatter(
        tsne_df.loc[mask, 'tsne1'],
        tsne_df.loc[mask, 'tsne2'],
        c=[palette[i]],
        label=cell_type,
        alpha=0.6,
        s=10,
        edgecolors='none'
    )

ax.set_xlabel('t-SNE 1', fontsize=14)
ax.set_ylabel('t-SNE 2', fontsize=14)
ax.set_title('t-SNE Visualization of Cell Type Embeddings', fontsize=16, fontweight='bold')

# Handle legend - if too many cell types, create a more compact legend
if n_types <= 30:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)
else:
    # For many cell types, save legend separately or skip
    print("Too many cell types for legend. Skipping legend in main plot.")

plt.tight_layout()
plt.savefig('tsne_cell_types.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'tsne_cell_types.png'")
plt.show()

# Optionally, create a version without labels but with density
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(
    tsne_df['tsne1'],
    tsne_df['tsne2'],
    c=pd.Categorical(tsne_df['cell_type']).codes,
    cmap='tab20' if n_types <= 20 else 'nipy_spectral',
    alpha=0.6,
    s=5,
    edgecolors='none'
)
ax.set_xlabel('t-SNE 1', fontsize=14)
ax.set_ylabel('t-SNE 2', fontsize=14)
ax.set_title('t-SNE Visualization (Colored by Cell Type)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('tsne_cell_types_no_legend.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'tsne_cell_types_no_legend.png'")
plt.show()

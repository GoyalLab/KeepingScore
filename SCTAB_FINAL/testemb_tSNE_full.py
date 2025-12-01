import anndata as ad
import pandas as pd
import scanpy as sc
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import seaborn as sns
import yaml

# Paths
DATA_PATH = '/projects/b1042/GoyalLab/jaekj/merlin_cxg_2023_05_15_sf-log1p'
YAML_PATH = '/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/scTab-devel/notebooks/data_augmentation/shortend_cell_types.yaml'

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

# Load cell type hierarchy
cell_type_hierarchy = np.load(join(DATA_PATH, 'cell_type_hierarchy/child_matrix.npy'))

# Load general tissue type
tissue_general = dd.read_parquet(join(DATA_PATH, 'test'), columns='tissue_general').compute().to_numpy()

# Load tissue type mapping
tissue_general_mapping = pd.read_parquet(join(DATA_PATH, 'categorical_lookup/tissue_general.parquet'))

# Load embeddings
print("Loading embeddings...")
test = torch.load("test_embedding.pt", map_location="cpu")

# Map test labels to cell type names
y_test_celltype_names = [idx_to_celltype[idx] for idx in y_test_names]

# Apply shortened names
y_test_display_names = [shortend_cell_types.get(name, name) for name in y_test_celltype_names]

# Convert embeddings to numpy if needed
# Extract embeddings from dictionary
# Common keys: 'embeddings', 'embedding', 'features', 'X'
print(f"Available keys in test dict: {test.keys()}")

# Try common key names
test_embeddings = test['X']
if isinstance(test_embeddings, torch.Tensor):
    test_embeddings = test_embeddings.numpy()

print(f"Embedding shape: {test_embeddings.shape}")
print(f"Number of labels: {len(y_test_display_names)}")


y_true = test["y"]

y_true = np.asarray(y_true).flatten()
tissue_general = np.asarray(tissue_general).flatten()

adata = ad.AnnData(
    X=test["X"].cpu().numpy(), 
    obs=pd.DataFrame({
        'y_true': cell_type_mapping.loc[y_true].to_numpy().flatten(), 
        'tissue': tissue_general_mapping.loc[tissue_general].to_numpy().flatten()
    })
)

adata.obs.y_true

# Without subsampling
# apply tSNE
sc.tl.tsne(adata, use_rep='X')

adata.write_h5ad('embedding_prediction_full.h5ad')

adata = ad.read_h5ad('embedding_prediction_full.h5ad')

plt.rcParams['figure.figsize'] = (5, 5)

cell_freq = adata.obs.y_true.value_counts()
# only plot most frequent cell types to not overload the color scale
cells_to_plot = cell_freq.index.tolist()
adata_plot = adata.copy()
adata_plot.obs['y_true'] = adata_plot.obs.y_true.mask(~adata_plot.obs.y_true.isin(cells_to_plot)).astype(str)

# convert to shortened cell type names
adata_plot.obs['y_true'] = (
    adata_plot.obs['y_true']
    .str.replace(r'[$]', '', regex=True)
    .str.replace(r'\\', '', regex=True)
)

n = adata_plot.obs["y_true"].nunique()
palette = sns.color_palette("hls", n)

adata_plot.obs["y_true"] = adata_plot.obs["y_true"].astype("category")

sc.pl.tsne(
    adata_plot,
    color="y_true",
    palette=palette,
    legend_fontsize='x-small',
    ncols=1,
    title='scTab test embedding',
    save='tSNE_test_embedding.png'
)

sc.pl.tsne(
    adata_plot,
    color="y_true",
    palette=palette,
    legend_fontsize='x-small',
    ncols=1,
    title='scTab test embedding',
    save='tSNE_test_embedding.svg'
)

unique_cell_types = adata_plot.obs["y_true"].unique()
print("Number of cell types in plot:", len(unique_cell_types))
print("Cell types:", unique_cell_types)

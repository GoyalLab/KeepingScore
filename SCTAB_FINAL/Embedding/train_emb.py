import sys
import torch
from os.path import join
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
import dask.dataframe as dd

sys.path.append("/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/scTab-devel/")
from cellnet.estimators import EstimatorCellTypeClassifier
from cellnet.models import TabnetClassifier
from notebooks.model_evaluation.utils import correct_labels

torch.set_float32_matmul_precision('high')

# Paths
DATA_PATH = '/projects/b1042/GoyalLab/jaekj/merlin_cxg_2023_05_15_sf-log1p'
CKPT_PATH = '/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/scTab-checkpoints/scTab/run4/val_f1_macro_epoch=45_val_f1_macro=0.847.ckpt'

cell_type_mapping = pd.read_parquet(join(DATA_PATH, 'categorical_lookup/cell_type.parquet'))
tissue_general_mapping = pd.read_parquet(join(DATA_PATH, 'categorical_lookup/tissue_general.parquet'))
cell_type_hierarchy = np.load(join(DATA_PATH, 'cell_type_hierarchy/child_matrix.npy'))

from natsort import natsorted

# Make sure the Series is categorical
cell_type_mapping['label'] = cell_type_mapping['label'].astype('category')

# Reorder categories using .cat
cell_type_mapping['label'] = cell_type_mapping['label'].cat.reorder_categories(
    natsorted(cell_type_mapping['label'].cat.categories)
)

# Make sure tissue_general column is categorical
tissue_general_mapping['label'] = tissue_general_mapping['label'].astype('category')

# Reorder categories using .cat
tissue_general_mapping['label'] = tissue_general_mapping['label'].cat.reorder_categories(
    natsorted(tissue_general_mapping['label'].cat.categories)
)

tissue_general = dd.read_parquet(join(DATA_PATH, 'train'), columns='tissue_general').compute().to_numpy().ravel()
y_true_train = dd.read_parquet(join(DATA_PATH, 'train'), columns='cell_type').compute().to_numpy().ravel()

# --- Load scTab estimator ---
estim = EstimatorCellTypeClassifier(DATA_PATH)
seed_everything(1)
estim.init_datamodule(batch_size=2048, dataloader_kwargs_train={"shuffle": False, "drop_last": False})

# --- Load TabNet model ---
estim.model = TabnetClassifier.load_from_checkpoint(
    CKPT_PATH, **estim.get_fixed_model_params('tabnet')
)
estim.trainer = pl.Trainer(logger=[], accelerator="cpu", devices=1)

probas = estim.predict(estim.datamodule.train_dataloader())
y_pred_train = np.argmax(probas, axis=1)

assert y_pred_train.shape[0] == y_true_train.shape[0]
y_pred_corr = correct_labels(y_true_train, y_pred_train, cell_type_hierarchy)

# --- Get feature embedding ---
estim.model.predict_bottleneck = True
X_train_emb = estim.predict(estim.datamodule.train_dataloader())
print("X_train_emb shape:", X_train_emb.shape)
estim.model.predict_bottleneck = False

# Convert to tensors & save
torch.save({
    "X": torch.from_numpy(X_train_emb).float(),
    "y_true": torch.from_numpy(y_true_train).long(),
    "y_pred": torch.from_numpy(y_pred_train).long()
}, "train_embedding.pt")
print("Saved train_embedding.pt")

# --- Save AnnData ---
import anndata
adata_train_emb = anndata.AnnData(
    X=X_train_emb,
    obs=pd.DataFrame({
        'y_true': cell_type_mapping.loc[y_true_train].to_numpy().flatten(),
        'y_pred': cell_type_mapping.loc[y_pred_corr].to_numpy().flatten(),
        'tissue_general': tissue_general_mapping.loc[tissue_general].to_numpy().flatten()
    })
)
adata_train_emb.obs['wrong_prediction'] = (adata_train_emb.obs.y_true != adata_train_emb.obs.y_pred).astype(str).astype('category')
adata_train_emb.obsm["X_emb"] = adata_train_emb.X

# Subsample, PCA, TSNE, UMAP
sc.pp.subsample(adata_train_emb, n_obs=200_000)
sc.pp.pca(adata_train_emb, n_comps=50)
sc.tl.tsne(adata_train_emb)
sc.pp.neighbors(adata_train_emb)
sc.tl.umap(adata_train_emb)

import os
os.makedirs("adatas", exist_ok=True)
adata_train_emb.write_h5ad('adatas/train_emb_tabnet.h5ad')


import matplotlib.pyplot as plt
import yaml
from cellnet.utils.cell_ontology import retrieve_child_nodes_from_ubergraph

plt.rcParams['figure.figsize'] = (5, 5)

# --- Only plot most frequent cell types to avoid color overload ---
cell_freq = adata_train_emb.obs.y_true.value_counts() 
cells_to_plot = cell_freq.index.tolist()[:70]

adata_plot = adata_train_emb.copy()  
# Add 'Other' as a valid category before replacement
for col in ['y_pred', 'y_true']:
    if isinstance(adata_plot.obs[col].dtype, pd.CategoricalDtype):
        adata_plot.obs[col] = adata_plot.obs[col].cat.add_categories(['Other'])

adata_plot.obs['y_pred'] = adata_plot.obs['y_pred'].where(
    adata_plot.obs['y_pred'].isin(cells_to_plot), other='Other'
)
adata_plot.obs['y_true'] = adata_plot.obs['y_true'].where(
    adata_plot.obs['y_true'].isin(cells_to_plot), other='Other'
)

# --- Replace with shortened cell type names ---
with open('/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/scTab-devel/notebooks/data_augmentation/shortend_cell_types.yaml', 'r') as f:
    shortend_cell_types = yaml.safe_load(f)

adata_plot.obs['y_pred'] = adata_plot.obs['y_pred'].replace(shortend_cell_types)
adata_plot.obs['y_true'] = adata_plot.obs['y_true'].replace(shortend_cell_types)

# --- t-SNE plot ---
ax = sc.pl.tsne(
    adata_plot, 
    color='y_true', 
    legend_fontsize='x-small', 
    ncols=1, 
    title='Learned features of scTab', 
    show=False
)
ax.get_legend().remove()
plt.savefig('adatas/train_tsne_learned_features.png')  

# --- Coarse cell type mapping ---
coarse_cell_types = [
    'neural cell','epithelial fate stem cell','endothelial cell',
    'epithelial cell of lung','oviduct glandular cell','respiratory epithelial cell',
    'duct epithelial cell','granulocyte','B cell','B cell, CD19-positive',
    'fibroblast','macrophage','monocyte','T cell','natural killer cell',
    'kidney cell','enterocyte','cardiac muscle cell','smooth muscle cell','plasma cell'
]

subtypes = retrieve_child_nodes_from_ubergraph(coarse_cell_types)
adata_train_emb.obs['coarse_cell_type'] = np.nan  
for cell_type in coarse_cell_types:
    if cell_type in subtypes:
        adata_train_emb.obs.loc[
            adata_train_emb.obs.y_true.isin(subtypes[cell_type]), 'coarse_cell_type'
        ] = cell_type
    else:
        print(f"Warning: {cell_type} not found in subtypes")

adata_train_emb.uns.pop('coarse_cell_type_colors', None)

# --- t-SNE using X_emb representation ---
sc.tl.tsne(adata_train_emb, use_rep="X_emb")
sc.pl.tsne(
    adata_train_emb,
    color='tissue_general',
    show=False
)
plt.savefig("adatas/train_emb_tsne_learned_features_coarse_cell_types.png", dpi=300, bbox_inches="tight")
plt.close()

# --- UMAP ---
sc.tl.umap(adata_train_emb)
sc.pl.umap(
    adata_train_emb,
    color='tissue_general',
    show=False
)
plt.savefig("adatas/train_emb_UMAP_learned_features_coarse_cell_types.png", dpi=300, bbox_inches="tight")
plt.close()

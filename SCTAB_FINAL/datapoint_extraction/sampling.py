#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import pandas as pd
from os.path import join
import argparse

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--mapping_path", type=str, required=True)
parser.add_argument("--sample_size", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

# ----------------------------
# Reproducibility
# ----------------------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ----------------------------
# Load embeddings
# ----------------------------
test_emb = torch.load(join(args.data_path, "test_embedding.pt"), weights_only=True)

X_test = test_emb["X"].cpu().numpy()
y_test = test_emb["y_true"].cpu().numpy()

# ----------------------------
# Load mapping
# ----------------------------
cell_type_mapping = pd.read_parquet(args.mapping_path)
label_to_name = cell_type_mapping["label"].to_dict()

# ----------------------------
# Sampling
# ----------------------------
unique_labels = np.unique(y_test)

X_list, y_list, idx_list, name_list = [], [], [], []

for label in unique_labels:
    class_indices = np.where(y_test == label)[0]

    if len(class_indices) > args.sample_size:
        selected = np.random.choice(class_indices, args.sample_size, replace=False)
    else:
        selected = class_indices

    X_list.append(X_test[selected])
    y_list.append(y_test[selected])
    idx_list.append(selected)
    name_list.extend([label_to_name[int(label)]] * len(selected))

# ----------------------------
# Concatenate
# ----------------------------
X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
idx_all = np.concatenate(idx_list, axis=0)
name_all = np.array(name_list, dtype=object)

# ----------------------------
# Save EVERYTHING in one file
# ----------------------------
np.savez_compressed(
    args.save_path,
    X=X_all,
    y=y_all,
    indices=idx_all,
    names=name_all
)

print(f"[✓] Saved to {args.save_path}")
print(f"Shape: {X_all.shape}")
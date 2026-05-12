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
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

# ----------------------------
# Load embeddings
# ----------------------------
test_emb = torch.load(join(args.data_path, "test_embedding.pt"), weights_only=True)

X_test = test_emb["X"].cpu().numpy()        # [N, D]
y_test = test_emb["y_true"].cpu().numpy()   # [N]

# ----------------------------
# Load label → name mapping
# ----------------------------
cell_type_mapping = pd.read_parquet(args.mapping_path)
label_to_name = cell_type_mapping["label"].to_dict()

print(f"[INFO] Loaded {len(label_to_name)} label names")

# ----------------------------
# Compute class averages
# ----------------------------
unique_labels = np.unique(y_test)

class_means = []
class_labels = []
class_names = []

for label in unique_labels:
    idx = np.where(y_test == label)[0]

    X_class = X_test[idx]           # [n_samples, D]
    mean_vec = X_class.mean(axis=0)  # [D]

    class_means.append(mean_vec)
    class_labels.append(label)
    class_names.append(label_to_name[int(label)])

    print(f"[INFO] Class {label}: {len(idx)} samples")

# ----------------------------
# Stack results
# ----------------------------
class_means = np.stack(class_means)        # [num_classes, D]
class_labels = np.array(class_labels)      # [num_classes]
class_names = np.array(class_names, dtype=object)

# ----------------------------
# Save (single file)
# ----------------------------
np.savez(
    args.save_path,
    X_mean=class_means,
    y=class_labels,
    names=class_names
)

print("\n[✓] Saved class averages")
print(f"Shape: {class_means.shape}")
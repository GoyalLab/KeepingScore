#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--z_test_path",
    type=str,
    default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/z_test.npy",
    help="Path to z_test.npy containing test embeddings with shape [N, D].",
)
parser.add_argument(
    "--y_test_path",
    type=str,
    default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/y_test.npy",
    help="Path to y_test.npy containing perturbation labels with shape [N].",
)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument(
    "--names_path",
    type=str,
    default="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/ExPert/anndata/splits/perturbation_names.npy",
    help="Path to perturbation_names.npy. Index i must be the name for integer label i.",
)
parser.add_argument(
    "--mapping_path",
    type=str,
    default=None,
    help="Deprecated fallback parquet with a 'label' column indexed by integer class id.",
)
args = parser.parse_args()

save_dir = os.path.dirname(args.save_path)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# Load test embeddings / labels
# ----------------------------
z_test = np.load(args.z_test_path).astype(np.float32, copy=False)
y_test = np.load(args.y_test_path).astype(np.int64, copy=False)

if z_test.ndim != 2:
    raise ValueError(f"Expected z_test to be 2D, got shape {z_test.shape}")
if y_test.ndim != 1:
    raise ValueError(f"Expected y_test to be 1D, got shape {y_test.shape}")
if z_test.shape[0] != y_test.shape[0]:
    raise ValueError(
        f"z_test rows ({z_test.shape[0]}) do not match y_test length ({y_test.shape[0]})"
    )

print(f"[INFO] Loaded z_test with shape {z_test.shape}")
print(f"[INFO] Loaded y_test with shape {y_test.shape}")

# ----------------------------
# Load label -> perturbation name mapping
# ----------------------------
if args.names_path:
    perturbation_names = np.load(args.names_path, allow_pickle=True).astype(str)
    label_to_name = {i: name for i, name in enumerate(perturbation_names)}
    print(f"[INFO] Loaded {len(label_to_name)} perturbation names from {args.names_path}")
elif args.mapping_path:
    mapping_df = pd.read_parquet(args.mapping_path)
    if "label" not in mapping_df.columns:
        raise KeyError(f"Expected a 'label' column in {args.mapping_path}")
    label_to_name = mapping_df["label"].to_dict()
    print(f"[INFO] Loaded {len(label_to_name)} label names from {args.mapping_path}")
else:
    raise ValueError("Provide --names_path or --mapping_path")

max_label = int(y_test.max())
missing_names = sorted(set(np.unique(y_test).astype(int)) - set(label_to_name))
if missing_names:
    raise ValueError(
        f"Labels in y_test are missing from the name mapping: {missing_names[:20]}"
    )
if max_label >= len(label_to_name):
    raise ValueError(
        f"y_test contains label {max_label}, but only {len(label_to_name)} names were loaded"
    )

control_labels = [
    label for label, name in label_to_name.items() if str(name).lower() == "control"
]
if control_labels:
    control_label = control_labels[0]
    control_count = int((y_test == control_label).sum())
    if control_count == 0:
        raise ValueError(
            f"'control' is present in the name mapping at label {control_label}, "
            "but no control rows are present in y_test"
        )
    print(f"[INFO] control label {control_label}: {control_count} test samples")

# ----------------------------
# Compute per-perturbation means
# ----------------------------
unique_labels = np.unique(y_test)

class_means = []
class_labels = []
class_names = []

for label in unique_labels:
    idx = np.where(y_test == label)[0]
    mean_vec = z_test[idx].mean(axis=0)

    class_means.append(mean_vec)
    class_labels.append(int(label))
    class_names.append(label_to_name.get(int(label), f"label_{int(label)}"))

    print(f"[INFO] Perturbation {int(label)}: {len(idx)} samples")

class_means = np.stack(class_means).astype(np.float32, copy=False)
class_labels = np.asarray(class_labels, dtype=np.int64)
class_names = np.asarray(class_names, dtype=object)

np.savez(
    args.save_path,
    X_mean=class_means,
    y=class_labels,
    names=class_names,
)

print("\n[✓] Saved perturbation means")
print(f"Shape: {class_means.shape}")

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
H5AD_PATH = BASE / "merged.h5ad"
OUT_DIR = BASE / "splits"
SPLIT_SEED = 42


def decode_strings(values):
    return np.array(
        [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in values]
    )


def read_categorical(handle, path):
    group = handle[path]
    codes = group["codes"][:]
    categories = decode_strings(group["categories"][:])
    return categories[codes]


def factorize_in_order(values):
    labels = np.empty(len(values), dtype=np.int64)
    uniques = []
    lookup = {}
    for i, value in enumerate(values):
        if value not in lookup:
            lookup[value] = len(uniques)
            uniques.append(value)
        labels[i] = lookup[value]
    return labels, np.array(uniques, dtype=object)


def allocate_control_splits(split, perturbation, dataset):
    rng = np.random.default_rng(SPLIT_SEED)
    split_for_export = pd.Series(split.copy())
    query_control_mask = (split_for_export == "query").to_numpy() & (
        perturbation == "control"
    )
    records = []

    for ds in pd.unique(dataset):
        dataset_mask = dataset == ds
        control_idx = np.flatnonzero(dataset_mask & query_control_mask)
        if len(control_idx) == 0:
            continue

        reference = pd.Series(split[dataset_mask & ~query_control_mask])
        split_counts = (
            reference[reference != "query"]
            .value_counts()
            .reindex(["train", "val", "test", "ood_test"], fill_value=0)
        )
        if split_counts.sum() == 0:
            raise RuntimeError(f"No non-query rows available for {ds}")

        expected = len(control_idx) * split_counts / split_counts.sum()
        target_counts = np.floor(expected).astype(int)
        remainder = int(len(control_idx) - target_counts.sum())
        if remainder:
            order = (expected - target_counts).sort_values(ascending=False).index[
                :remainder
            ]
            target_counts.loc[order] += 1

        shuffled = rng.permutation(control_idx)
        start = 0
        for split_name, count in target_counts.items():
            count = int(count)
            if count == 0:
                continue
            chosen = shuffled[start : start + count]
            split_for_export.iloc[chosen] = split_name
            records.append(
                {"dataset": ds, "split": split_name, "n_controls": int(count)}
            )
            start += count

    summary = pd.DataFrame(records).pivot_table(
        index="dataset",
        columns="split",
        values="n_controls",
        fill_value=0,
        aggfunc="sum",
    )
    summary = summary.reindex(
        columns=["train", "val", "test", "ood_test"], fill_value=0
    )
    return split_for_export.to_numpy(dtype=object), summary


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with h5py.File(H5AD_PATH, "r") as handle:
        split = read_categorical(handle, "obs/split")
        perturbation = read_categorical(handle, "obs/perturbation")
        dataset = read_categorical(handle, "obs/dataset")
        latent = handle["obsm/ExPert_latent_z_shared"][:]

    effective_split, summary = allocate_control_splits(split, perturbation, dataset)
    labels, uniques = factorize_in_order(perturbation)

    split_masks = {
        "train": effective_split == "train",
        "val": effective_split == "val",
        "test": effective_split == "test",
        "ood": effective_split == "ood_test",
    }
    for name, mask in split_masks.items():
        np.save(OUT_DIR / f"z_{name}.npy", latent[mask])
        np.save(OUT_DIR / f"y_{name}.npy", labels[mask])

    np.save(OUT_DIR / "perturbation_names.npy", uniques)
    np.save(OUT_DIR / "effective_split.npy", effective_split)
    summary.to_csv(OUT_DIR / "control_allocation_summary.csv")

    split_counts = (
        pd.Series(effective_split)
        .value_counts()
        .reindex(["train", "val", "test", "ood_test", "query"], fill_value=0)
    )
    print(split_counts.to_string())
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()

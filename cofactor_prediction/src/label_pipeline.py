from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .manifests import DatasetManifest, ensure_dir, read_json, write_json, utc_now_iso
from .pickle_compat import install_numpy_pickle_compat_aliases


NO_COFACTOR = "no_cofactor"


def _load_vocab_order(vocab_json: Path) -> List[str]:
    with vocab_json.open("r") as f:
        obj = json.load(f)
    return [str(cf) for cf in obj["cofactors"]]


def _build_split_samples(
    split_name: str,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    include_no_cofactor: bool,
) -> List[Dict]:
    samples: List[Dict] = []

    for _, row in pos_df.iterrows():
        pid = str(row["protein_id"])
        emb = str(row["embedding_path"])
        cofactors = row.get("cofactors", [])
        cofactors = [str(c) for c in cofactors] if isinstance(cofactors, list) else []

        if not cofactors and include_no_cofactor:
            samples.append(
                {
                    "split": split_name,
                    "protein_id": pid,
                    "label_name": NO_COFACTOR,
                    "embedding_path": emb,
                    "source": "positive_empty",
                }
            )
        else:
            for cf in sorted(set(cofactors)):
                samples.append(
                    {
                        "split": split_name,
                        "protein_id": pid,
                        "label_name": cf,
                        "embedding_path": emb,
                        "source": "positive",
                    }
                )

    if include_no_cofactor:
        for _, row in neg_df.iterrows():
            samples.append(
                {
                    "split": split_name,
                    "protein_id": str(row["protein_id"]),
                    "label_name": NO_COFACTOR,
                    "embedding_path": str(row["embedding_path"]),
                    "source": "negative",
                }
            )

    return samples


def _load_embeddings(sample_rows: List[Dict]) -> np.ndarray:
    cache: Dict[str, np.ndarray] = {}
    vectors: List[np.ndarray] = []
    for row in sample_rows:
        emb_path = str(row["embedding_path"])
        if emb_path not in cache:
            vec = torch.load(emb_path, map_location="cpu")
            if isinstance(vec, torch.Tensor):
                vec_np = vec.detach().cpu().numpy().astype(np.float32)
            else:
                vec_np = np.asarray(vec, dtype=np.float32)
            if vec_np.ndim != 1:
                raise ValueError(f"Embedding at {emb_path} is not 1D, got {vec_np.shape}")
            cache[emb_path] = vec_np
        vectors.append(cache[emb_path])
    return np.stack(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)


def prepare_dataset(
    split_manifest_path: Path,
    positive_inventory_pkl: Path,
    negative_inventory_pkl: Path,
    vocab_json: Path,
    out_dir: Path,
    min_train_support: int = 5,
    include_no_cofactor: bool = True,
) -> Path:
    ensure_dir(out_dir)
    install_numpy_pickle_compat_aliases()

    split_manifest = read_json(split_manifest_path)
    positive_inventory = pd.read_pickle(positive_inventory_pkl)
    negative_inventory = pd.read_pickle(negative_inventory_pkl)

    pos_by_pid = positive_inventory.set_index("protein_id", drop=False)
    neg_by_pid = negative_inventory.set_index("protein_id", drop=False)

    def _slice_by_ids(df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
        if not ids:
            return df.iloc[0:0].copy()
        subset = df.reindex(ids)
        subset = subset[subset["protein_id"].notna()].copy()
        subset["protein_id"] = subset["protein_id"].astype(str)
        return subset.reset_index(drop=True)

    raw_samples_by_split: Dict[str, List[Dict]] = {}
    for split in ["train", "dev", "test"]:
        pos_ids = split_manifest["positive"][split]
        neg_ids = split_manifest["negative"][split]

        pos_df = _slice_by_ids(pos_by_pid, [str(x) for x in pos_ids])
        neg_df = _slice_by_ids(neg_by_pid, [str(x) for x in neg_ids])

        raw_samples_by_split[split] = _build_split_samples(split, pos_df, neg_df, include_no_cofactor=include_no_cofactor)

    train_counts = Counter([r["label_name"] for r in raw_samples_by_split["train"]])
    vocab_order = _load_vocab_order(vocab_json)

    valid_labels: List[str] = []
    if include_no_cofactor:
        valid_labels.append(NO_COFACTOR)
    for cf in vocab_order:
        if train_counts.get(cf, 0) >= min_train_support:
            valid_labels.append(cf)

    dropped = {cf: int(train_counts.get(cf, 0)) for cf in vocab_order if cf not in valid_labels and train_counts.get(cf, 0) > 0}

    class_to_idx = {name: idx for idx, name in enumerate(valid_labels)}
    idx_to_class = {str(idx): name for name, idx in class_to_idx.items()}

    split_files = {}
    split_counts = {}
    class_support: Dict[str, Dict[str, int]] = {}

    for split in ["train", "dev", "test"]:
        filtered = [r for r in raw_samples_by_split[split] if r["label_name"] in class_to_idx]
        if split in ("dev", "test") and len(filtered) == 0:
            raise RuntimeError(f"{split} split is empty after label filtering")

        labels = np.array([class_to_idx[r["label_name"]] for r in filtered], dtype=np.int64)
        protein_ids = np.array([r["protein_id"] for r in filtered], dtype=object)
        label_names = np.array([r["label_name"] for r in filtered], dtype=object)
        embeddings = _load_embeddings(filtered)

        split_npz = out_dir / f"{split}.npz"
        np.savez(
            split_npz,
            embeddings=embeddings,
            labels=labels,
            protein_ids=protein_ids,
            label_names=label_names,
        )

        pd.DataFrame(filtered).to_csv(out_dir / f"samples_{split}.csv", index=False)

        split_files[split] = str(split_npz)
        split_counts[split] = int(len(filtered))
        split_counter = Counter(label_names.tolist())
        class_support[split] = {name: int(split_counter.get(name, 0)) for name in valid_labels}

    report = {
        "split_manifest_path": str(split_manifest_path),
        "min_train_support": int(min_train_support),
        "include_no_cofactor": bool(include_no_cofactor),
        "train_raw_support": {k: int(v) for k, v in sorted(train_counts.items())},
        "valid_labels": valid_labels,
        "dropped_labels": dropped,
        "split_counts": split_counts,
        "class_support": class_support,
    }
    write_json(out_dir / "class_filter_report.json", report)

    dataset_manifest = DatasetManifest(
        split_type=split_manifest["split_type"],
        split_manifest_path=str(split_manifest_path),
        include_no_cofactor=include_no_cofactor,
        min_train_support=min_train_support,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        created_at=utc_now_iso(),
        split_files=split_files,
        split_counts=split_counts,
        class_support=class_support,
        dropped_classes={k: int(v) for k, v in dropped.items()},
    )
    dataset_manifest_path = out_dir / "dataset_manifest.json"
    dataset_manifest.write(dataset_manifest_path)
    return dataset_manifest_path

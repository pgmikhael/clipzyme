from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


REQUIRED_PREDICTION_KEYS = ["probs", "labels", "pred_labels", "protein_ids", "class_names"]


@dataclass
class PredictionBundle:
    probs: np.ndarray
    labels: np.ndarray
    pred_labels: np.ndarray
    protein_ids: np.ndarray
    class_names: np.ndarray


@dataclass
class DatasetBundle:
    embeddings: np.ndarray
    labels: np.ndarray
    protein_ids: np.ndarray
    label_names: np.ndarray


class ContractError(RuntimeError):
    """Raised when an artifact violates an expected I/O contract."""


def validate_prediction_npz(path: Path) -> None:
    if not path.exists():
        raise ContractError(f"Prediction file does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED_PREDICTION_KEYS if k not in data]
    if missing:
        raise ContractError(f"Prediction file {path} missing keys: {missing}")

    probs = data["probs"]
    labels = data["labels"]
    pred_labels = data["pred_labels"]
    protein_ids = data["protein_ids"]
    class_names = data["class_names"]

    if probs.ndim != 2:
        raise ContractError(f"probs must be 2D [N,C], got shape {probs.shape}")
    n = probs.shape[0]
    c = probs.shape[1]
    if labels.shape[0] != n or pred_labels.shape[0] != n or protein_ids.shape[0] != n:
        raise ContractError("labels/pred_labels/protein_ids length mismatch against probs")
    if class_names.shape[0] != c:
        raise ContractError("class_names length mismatch against probs class dimension")


def validate_dataset_npz(path: Path) -> None:
    if not path.exists():
        raise ContractError(f"Dataset split file does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    for key in ["embeddings", "labels", "protein_ids", "label_names"]:
        if key not in data:
            raise ContractError(f"Dataset split file {path} missing key: {key}")
    emb = data["embeddings"]
    labels = data["labels"]
    pids = data["protein_ids"]
    names = data["label_names"]
    if emb.ndim != 2:
        raise ContractError(f"embeddings must be 2D [N,D], got {emb.shape}")
    n = emb.shape[0]
    if labels.shape[0] != n or pids.shape[0] != n or names.shape[0] != n:
        raise ContractError("Dataset split arrays do not align by sample axis")


def check_disjoint(sets: Dict[str, Iterable[str]]) -> Dict[str, int]:
    names = list(sets.keys())
    out: Dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = set(sets[a]) & set(sets[b])
            out[f"{a}_intersect_{b}"] = len(inter)
    return out


def class_support(labels: Sequence[int], idx_to_class: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in idx_to_class.values()}
    for y in labels:
        name = idx_to_class[str(int(y))]
        counts[name] += 1
    return counts

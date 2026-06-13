from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.label_pipeline import NO_COFACTOR, prepare_dataset


def _write_embedding(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(values, dtype=torch.float32), path)


def test_label_pipeline_expansion_and_filtering(tmp_path: Path) -> None:
    emb_dir = tmp_path / "emb"

    p1 = emb_dir / "p1.pt"
    p2 = emb_dir / "p2.pt"
    p3 = emb_dir / "p3.pt"
    p4 = emb_dir / "p4.pt"
    n1 = emb_dir / "n1.pt"
    n2 = emb_dir / "n2.pt"

    _write_embedding(p1, [1, 0, 0, 0])
    _write_embedding(p2, [0, 1, 0, 0])
    _write_embedding(p3, [0, 0, 1, 0])
    _write_embedding(p4, [0, 0, 0, 1])
    _write_embedding(n1, [1, 1, 0, 0])
    _write_embedding(n2, [0, 1, 1, 0])

    pos = pd.DataFrame(
        [
            {"protein_id": "p1", "cofactors": ["cfA", "cfB"], "embedding_path": str(p1)},
            {"protein_id": "p2", "cofactors": ["cfA"], "embedding_path": str(p2)},
            {"protein_id": "p3", "cofactors": ["cfC"], "embedding_path": str(p3)},
            {"protein_id": "p4", "cofactors": [], "embedding_path": str(p4)},
        ]
    )
    neg = pd.DataFrame(
        [
            {"protein_id": "n1", "cofactors": [], "embedding_path": str(n1)},
            {"protein_id": "n2", "cofactors": [], "embedding_path": str(n2)},
        ]
    )

    pos_pkl = tmp_path / "positive_inventory.pkl"
    neg_pkl = tmp_path / "negative_inventory.pkl"
    pos.to_pickle(pos_pkl)
    neg.to_pickle(neg_pkl)

    split_manifest = {
        "split_type": "random_disjoint",
        "positive": {
            "train": ["p1", "p2", "p3"],
            "dev": ["p4"],
            "test": ["p1"],
        },
        "negative": {
            "train": ["n1"],
            "dev": [],
            "test": ["n2"],
        },
    }
    split_manifest_path = tmp_path / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest))

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"cofactors": ["cfA", "cfB", "cfC"]}))

    out_dir = tmp_path / "dataset"
    dataset_manifest_path = prepare_dataset(
        split_manifest_path=split_manifest_path,
        positive_inventory_pkl=pos_pkl,
        negative_inventory_pkl=neg_pkl,
        vocab_json=vocab_path,
        out_dir=out_dir,
        min_train_support=2,
        include_no_cofactor=True,
    )

    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    assert dataset_manifest["class_to_idx"][NO_COFACTOR] == 0
    assert dataset_manifest["class_to_idx"]["cfA"] == 1
    assert "cfB" not in dataset_manifest["class_to_idx"]
    assert "cfC" not in dataset_manifest["class_to_idx"]

    train_npz = np.load(out_dir / "train.npz", allow_pickle=True)
    dev_npz = np.load(out_dir / "dev.npz", allow_pickle=True)
    test_npz = np.load(out_dir / "test.npz", allow_pickle=True)

    assert train_npz["labels"].shape[0] == 3
    assert dev_npz["labels"].shape[0] == 1
    assert test_npz["labels"].shape[0] == 2

    train_labels = train_npz["labels"].tolist()
    assert train_labels.count(1) == 2
    assert train_labels.count(0) == 1

    report = json.loads((out_dir / "class_filter_report.json").read_text())
    assert "cfB" in report["dropped_labels"]
    assert "cfC" in report["dropped_labels"]

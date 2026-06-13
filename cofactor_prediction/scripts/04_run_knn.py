#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys
from typing import List

USER_SITE = str(site.getusersitepackages())
HOME_DIR = str(Path.home())
sys.path[:] = [
    p
    for p in sys.path
    if p != USER_SITE and not (p.startswith(HOME_DIR) and ".local" in p and "site-packages" in p)
]

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_contracts import validate_dataset_npz, validate_prediction_npz
from src.manifests import RunManifest, ensure_dir, read_json, sha256_file
from src.models_knn import run_knn_sweep


def _parse_k_list(raw: str) -> List[int]:
    ks = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        ks.append(k)
    if not ks:
        raise ValueError("k list is empty")
    return sorted(set(ks))


def _class_names_from_manifest(dataset_manifest: dict) -> List[str]:
    idx_to_class = dataset_manifest["idx_to_class"]
    return [idx_to_class[str(i)] for i in range(len(idx_to_class))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KNN cofactor prediction sweep on prepared dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="Path to dataset_manifest.json")
    p.add_argument("--k-list", type=str, default="1,3,5,10,20,50")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    dataset_manifest = read_json(args.dataset)
    split_files = dataset_manifest["split_files"]
    train_path = Path(split_files["train"])
    test_path = Path(split_files["test"])
    dev_path = Path(split_files["dev"])

    validate_dataset_npz(train_path)
    validate_dataset_npz(dev_path)
    validate_dataset_npz(test_path)

    train_npz = np.load(train_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)

    train_x = train_npz["embeddings"].astype(np.float32)
    train_y = train_npz["labels"].astype(np.int64)
    test_x = test_npz["embeddings"].astype(np.float32)
    test_y = test_npz["labels"].astype(np.int64)
    test_protein_ids = test_npz["protein_ids"]

    class_names = _class_names_from_manifest(dataset_manifest)
    num_classes = len(class_names)
    k_list = _parse_k_list(args.k_list)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(train_x.shape[0])
    train_x = train_x[perm]
    train_y = train_y[perm]

    probs_by_k = run_knn_sweep(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        k_list=k_list,
        num_classes=num_classes,
    )

    dataset_checksums = {
        "train": sha256_file(train_path),
        "dev": sha256_file(dev_path),
        "test": sha256_file(test_path),
        "dataset_manifest": sha256_file(args.dataset),
    }

    for k in k_list:
        model_name = f"knn_k{k}"
        run_id = f"{dataset_manifest['split_type']}__seed{args.seed}__{model_name}"
        run_dir = args.out_dir / run_id
        ensure_dir(run_dir)

        probs = probs_by_k[k]
        pred_labels = np.argmax(probs, axis=1).astype(np.int64)

        pred_path = run_dir / "predictions.npz"
        np.savez(
            pred_path,
            probs=probs,
            labels=test_y,
            pred_labels=pred_labels,
            protein_ids=test_protein_ids,
            class_names=np.array(class_names, dtype=object),
        )
        validate_prediction_npz(pred_path)

        run_manifest = RunManifest(
            run_id=run_id,
            split_type=dataset_manifest["split_type"],
            seed=int(args.seed),
            model_family="knn",
            model_name=model_name,
            dataset_manifest_path=str(args.dataset),
            config={
                "k": int(k),
                "k_list": k_list,
                "distance": "cosine",
                "voting": "similarity_weighted",
                "normalized_embeddings": True,
                "class_to_idx": dataset_manifest["class_to_idx"],
                "idx_to_class": dataset_manifest["idx_to_class"],
                "dataset_checksums": dataset_checksums,
            },
            artifacts={"predictions": str(pred_path)},
        )
        run_manifest.write(run_dir / "run_manifest.json")

        print(f"Saved KNN run: {run_dir}")


if __name__ == "__main__":
    main()

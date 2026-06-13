#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys
from typing import Dict

USER_SITE = str(site.getusersitepackages())
HOME_DIR = str(Path.home())
sys.path[:] = [
    p
    for p in sys.path
    if p != USER_SITE and not (p.startswith(HOME_DIR) and ".local" in p and "site-packages" in p)
]

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_contracts import validate_dataset_npz, validate_prediction_npz
from src.manifests import RunManifest, ensure_dir, read_json, sha256_file
from src.models_mlp import MLPConfig, train_mlp


def _class_names_from_manifest(dataset_manifest: dict):
    idx_to_class = dataset_manifest["idx_to_class"]
    return [idx_to_class[str(i)] for i in range(len(idx_to_class))]


def _load_config(config_path: Path, config_name: str) -> MLPConfig:
    with config_path.open("r") as f:
        payload = yaml.safe_load(f) or {}

    if "mlp_configs" in payload:
        cfg_table = payload["mlp_configs"]
    else:
        cfg_table = payload

    if config_name not in cfg_table:
        available = sorted(cfg_table.keys())
        raise KeyError(f"Config '{config_name}' not found in {config_path}. Available: {available}")

    return MLPConfig.from_dict(config_name, cfg_table[config_name])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MLP cofactor prediction model on prepared dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="Path to dataset_manifest.json")
    p.add_argument("--config", type=Path, required=True, help="YAML containing mlp_configs")
    p.add_argument("--config-name", type=str, required=True, help="Key under mlp_configs")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    dataset_manifest = read_json(args.dataset)
    split_files = dataset_manifest["split_files"]
    train_path = Path(split_files["train"])
    dev_path = Path(split_files["dev"])
    test_path = Path(split_files["test"])

    validate_dataset_npz(train_path)
    validate_dataset_npz(dev_path)
    validate_dataset_npz(test_path)

    train_npz = np.load(train_path, allow_pickle=True)
    dev_npz = np.load(dev_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)

    train_x = train_npz["embeddings"].astype(np.float32)
    train_y = train_npz["labels"].astype(np.int64)
    dev_x = dev_npz["embeddings"].astype(np.float32)
    dev_y = dev_npz["labels"].astype(np.int64)
    test_x = test_npz["embeddings"].astype(np.float32)
    test_y = test_npz["labels"].astype(np.int64)
    test_protein_ids = test_npz["protein_ids"]

    class_names = _class_names_from_manifest(dataset_manifest)
    cfg = _load_config(args.config, args.config_name)

    result = train_mlp(
        train_x=train_x,
        train_y=train_y,
        dev_x=dev_x,
        dev_y=dev_y,
        test_x=test_x,
        test_y=test_y,
        config=cfg,
        seed=args.seed,
    )

    model_name = cfg.name
    run_id = f"{dataset_manifest['split_type']}__seed{args.seed}__{model_name}"
    run_dir = args.out_dir / run_id
    ensure_dir(run_dir)

    pred_path = run_dir / "predictions.npz"
    np.savez(
        pred_path,
        probs=result["probs"],
        labels=result["labels"],
        pred_labels=result["pred_labels"],
        protein_ids=test_protein_ids,
        class_names=np.array(class_names, dtype=object),
    )
    validate_prediction_npz(pred_path)

    history_df = pd.DataFrame(result["history"])
    history_path = run_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    checkpoint_path = run_dir / "best_model.pt"
    torch.save(result["state_dict"], checkpoint_path)

    dataset_checksums: Dict[str, str] = {
        "train": sha256_file(train_path),
        "dev": sha256_file(dev_path),
        "test": sha256_file(test_path),
        "dataset_manifest": sha256_file(args.dataset),
        "config_file": sha256_file(args.config),
    }

    run_manifest = RunManifest(
        run_id=run_id,
        split_type=dataset_manifest["split_type"],
        seed=int(args.seed),
        model_family="mlp",
        model_name=model_name,
        dataset_manifest_path=str(args.dataset),
        config={
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "loss_type": cfg.loss_type,
            "use_class_weights": cfg.use_class_weights,
            "focal_gamma": cfg.focal_gamma,
            "best_epoch": int(result["best_epoch"]),
            "best_dev_macro_f1": float(result["best_dev_macro_f1"]),
            "class_weight": result["class_weight"],
            "class_to_idx": dataset_manifest["class_to_idx"],
            "idx_to_class": dataset_manifest["idx_to_class"],
            "dataset_checksums": dataset_checksums,
        },
        artifacts={
            "predictions": str(pred_path),
            "training_history": str(history_path),
            "best_model": str(checkpoint_path),
        },
    )
    run_manifest.write(run_dir / "run_manifest.json")

    print(f"Saved MLP run: {run_dir}")


if __name__ == "__main__":
    main()

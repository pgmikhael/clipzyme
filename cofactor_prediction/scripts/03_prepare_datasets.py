#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_inventory import load_paths_config
from src.label_pipeline import prepare_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare train/dev/test NPZ datasets for a split manifest.")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--inventory-dir", type=Path, required=True)
    p.add_argument("--paths-config", type=Path, default=Path("cofactor_prediction/configs/paths.yaml"))
    p.add_argument("--min-train-support", type=int, default=5)
    p.add_argument(
        "--include-no-cofactor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include no_cofactor as class 0.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    paths = load_paths_config(args.paths_config, root=root)

    dataset_manifest_path = prepare_dataset(
        split_manifest_path=args.split_manifest,
        positive_inventory_pkl=args.inventory_dir / "positive_inventory.pkl",
        negative_inventory_pkl=args.inventory_dir / "negative_inventory.pkl",
        vocab_json=paths["vocab_json"],
        out_dir=args.out_dir,
        min_train_support=args.min_train_support,
        include_no_cofactor=args.include_no_cofactor,
    )

    print(f"Saved dataset manifest: {dataset_manifest_path}")


if __name__ == "__main__":
    main()

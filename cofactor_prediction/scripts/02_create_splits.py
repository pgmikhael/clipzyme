#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys

USER_SITE = str(site.getusersitepackages())
HOME_DIR = str(Path.home())
sys.path[:] = [
    p
    for p in sys.path
    if p != USER_SITE and not (p.startswith(HOME_DIR) and ".local" in p and "site-packages" in p)
]

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_inventory import load_paths_config
from src.pickle_compat import install_numpy_pickle_compat_aliases
from src.split_builders import (
    create_balanced_rule_disjoint_v3,
    create_legacy_original_split,
    create_random_disjoint_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create split manifests for cofactor prediction.")
    p.add_argument(
        "--split-type",
        type=str,
        required=True,
        choices=["legacy_original", "random_disjoint", "balanced_rule_disjoint_v3"],
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--inventory-dir", type=Path, default=Path("cofactor_prediction/runs/inventory"))
    p.add_argument("--paths-config", type=Path, default=Path("cofactor_prediction/configs/paths.yaml"))
    p.add_argument("--min-train-support", type=int, default=5)
    p.add_argument("--min-test-support", type=int, default=1)
    p.add_argument("--n-restarts", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.out_dir / f"{args.split_type}_seed{args.seed}.json"

    if args.split_type == "legacy_original":
        root = Path.cwd()
        paths = load_paths_config(args.paths_config, root=root)
        manifest = create_legacy_original_split(
            positive_train_pkl=paths["positive_train_pkl"],
            positive_dev_pkl=paths["positive_dev_pkl"],
            positive_test_pkl=paths["positive_test_pkl"],
            negative_train_csv=paths["negative_train_csv"],
            negative_dev_csv=paths["negative_dev_csv"],
            negative_test_csv=paths["negative_test_csv"],
            output_path=output_manifest,
            seed=args.seed,
            min_train_support=args.min_train_support,
            min_test_support=args.min_test_support,
        )
    else:
        install_numpy_pickle_compat_aliases()
        pos = pd.read_pickle(args.inventory_dir / "positive_inventory.pkl")
        neg = pd.read_pickle(args.inventory_dir / "negative_inventory.pkl")

        if args.split_type == "random_disjoint":
            manifest = create_random_disjoint_split(
                positive_inventory=pos,
                negative_inventory=neg,
                output_path=output_manifest,
                seed=args.seed,
                min_train_support=args.min_train_support,
                min_test_support=args.min_test_support,
            )
        else:
            manifest = create_balanced_rule_disjoint_v3(
                positive_inventory=pos,
                negative_inventory=neg,
                output_path=output_manifest,
                seed=args.seed,
                min_train_support=args.min_train_support,
                min_test_support=args.min_test_support,
                n_restarts=args.n_restarts,
            )

    print(f"Saved split manifest: {output_manifest}")
    print(f"Split type: {manifest.split_type}")
    print(f"Diagnostics: n_testable={manifest.diagnostics.get('n_testable')}")


if __name__ == "__main__":
    main()

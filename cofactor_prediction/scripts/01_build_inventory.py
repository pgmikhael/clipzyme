#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_inventory import build_inventory, load_paths_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build globally deduplicated cofactor inventory.")
    p.add_argument("--paths-config", type=Path, default=Path("cofactor_prediction/configs/paths.yaml"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--drop-missing-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop proteins without available embedding files (recommended).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    paths = load_paths_config(args.paths_config, root=root)
    artifacts = build_inventory(
        paths=paths,
        output_dir=args.out_dir,
        root=root,
        drop_missing_embeddings=args.drop_missing_embeddings,
    )
    print("Inventory complete")
    print(f"  positive: {artifacts.positive_inventory_pkl}")
    print(f"  negative: {artifacts.negative_inventory_pkl}")
    print(f"  summary:  {artifacts.summary_json}")


if __name__ == "__main__":
    main()

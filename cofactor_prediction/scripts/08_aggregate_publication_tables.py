#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.aggregation import aggregate_publication_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate run artifacts into publication-ready tables and figures.")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    index_payload = aggregate_publication_outputs(run_root=args.run_root, out_dir=args.out_dir)
    print(f"Saved publication manifest: {args.out_dir / 'manifests' / 'experiment_index.json'}")
    print(f"Aggregated runs: {index_payload['n_run_dirs']}")


if __name__ == "__main__":
    main()

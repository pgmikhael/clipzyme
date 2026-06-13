#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys
from typing import Optional

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

from src.io_contracts import validate_prediction_npz
from src.manifests import read_json, write_json
from src.roc import generate_roc_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ROC artifacts from predictions.npz")
    p.add_argument("--predictions", type=Path, default=None, help="Path to predictions.npz")
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory containing predictions.npz")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _resolve_predictions(run_dir: Optional[Path], predictions: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir / "predictions.npz"
    if predictions is not None:
        return predictions
    raise ValueError("Provide --run-dir or --predictions")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    pred_path = _resolve_predictions(run_dir, args.predictions)
    validate_prediction_npz(pred_path)

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif run_dir is not None:
        out_dir = run_dir / "roc"
    else:
        out_dir = pred_path.parent / "roc"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(pred_path, allow_pickle=True)
    probs = pred["probs"]
    labels = pred["labels"].astype(np.int64)
    class_names = [str(x) for x in pred["class_names"].tolist()]

    summary = generate_roc_artifacts(probs=probs, labels=labels, class_names=class_names, out_dir=out_dir)

    if run_dir is not None:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            run_manifest = read_json(manifest_path)
            artifacts = dict(run_manifest.get("artifacts", {}))
            artifacts.update(
                {
                    "roc_summary": str(out_dir / "roc_summary.json"),
                    "roc_points": summary["roc_points_path"],
                    "roc_overall_png": summary["plot_paths"]["overall"],
                    "roc_per_class_png": summary["plot_paths"]["per_class"],
                    "roc_bar_png": summary["plot_paths"]["bar"],
                }
            )
            run_manifest["artifacts"] = artifacts
            write_json(manifest_path, run_manifest)

    print(f"Saved ROC summary: {out_dir / 'roc_summary.json'}")


if __name__ == "__main__":
    main()

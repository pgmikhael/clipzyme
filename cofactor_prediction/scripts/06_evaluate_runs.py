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
from src.metrics import compute_classification_metrics, save_confusion_matrix_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a predictions artifact and save metrics.")
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
        out_dir = run_dir
    else:
        out_dir = pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(pred_path, allow_pickle=True)
    probs = pred["probs"]
    labels = pred["labels"].astype(np.int64)
    class_names = [str(x) for x in pred["class_names"].tolist()]

    metrics = compute_classification_metrics(probs=probs, labels=labels, class_names=class_names)

    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics)

    confusion_csv = out_dir / "confusion_matrix.csv"
    save_confusion_matrix_csv(metrics, confusion_csv)

    if run_dir is not None:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            run_manifest = read_json(manifest_path)
            artifacts = dict(run_manifest.get("artifacts", {}))
            artifacts.update(
                {
                    "metrics": str(metrics_path),
                    "confusion_matrix_csv": str(confusion_csv),
                }
            )
            run_manifest["artifacts"] = artifacts
            write_json(manifest_path, run_manifest)

    print(f"Saved metrics: {metrics_path}")
    print(f"macro_f1={metrics['macro_f1']:.6f} macro_auroc={metrics['macro_auroc']:.6f} micro_auroc={metrics['micro_auroc']:.6f}")


if __name__ == "__main__":
    main()

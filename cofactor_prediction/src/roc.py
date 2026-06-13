from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from .manifests import write_json


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.int32)
    y[np.arange(labels.shape[0]), labels.astype(int)] = 1
    return y


def _save_roc_points(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, str]:
    parquet_path = out_dir / "roc_points.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path, "parquet"
    except Exception:
        csv_path = out_dir / "roc_points.csv"
        df.to_csv(csv_path, index=False)
        return csv_path, "csv_fallback"


def _plot_overall(
    out_path: Path,
    micro: Tuple[np.ndarray, np.ndarray, float],
    macro: Tuple[np.ndarray, np.ndarray, float],
    weighted: Tuple[np.ndarray, np.ndarray, float],
) -> None:
    plt.figure(figsize=(10, 8))
    plt.plot(micro[0], micro[1], label=f"Micro (AUC={micro[2]:.3f})", lw=2)
    plt.plot(macro[0], macro[1], label=f"Macro (AUC={macro[2]:.3f})", lw=2)
    plt.plot(weighted[0], weighted[1], label=f"Weighted (AUC={weighted[2]:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC (Micro/Macro/Weighted)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_per_class(out_path: Path, curves: Dict[str, Tuple[np.ndarray, np.ndarray, float, int]]) -> None:
    plt.figure(figsize=(12, 10))
    for name, (fpr, tpr, roc_auc, support) in sorted(curves.items(), key=lambda kv: kv[1][2], reverse=True):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={roc_auc:.3f}, n={support})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-Class ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_bar(out_path: Path, curves: Dict[str, Tuple[np.ndarray, np.ndarray, float, int]]) -> None:
    items = sorted(((name, vals[2], vals[3]) for name, vals in curves.items()), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in items]
    aucs = [x[1] for x in items]
    supports = [x[2] for x in items]

    plt.figure(figsize=(14, 7))
    idx = np.arange(len(items))
    bars = plt.bar(idx, aucs)
    plt.axhline(0.5, color="red", linestyle="--", lw=1)
    plt.xticks(idx, names, rotation=45, ha="right", fontsize=9)
    plt.ylabel("AUROC")
    plt.title("Per-Class AUROC")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    for bar, auc_value, support in zip(bars, aucs, supports):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{auc_value:.2f}\n(n={support})", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_roc_artifacts(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = labels.astype(int)
    num_classes = probs.shape[1]
    y_true = _one_hot(labels, num_classes)

    per_class_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float, int]] = {}
    records = []

    for i, name in enumerate(class_names):
        y = y_true[:, i]
        positives = int(y.sum())
        negatives = int((1 - y).sum())
        if positives == 0 or negatives == 0:
            continue
        fpr, tpr, thresholds = roc_curve(y, probs[:, i])
        roc_auc = float(auc(fpr, tpr))
        per_class_curves[name] = (fpr, tpr, roc_auc, positives)
        for fp, tp, th in zip(fpr, tpr, thresholds):
            records.append(
                {
                    "curve_type": "class",
                    "class_name": name,
                    "fpr": float(fp),
                    "tpr": float(tp),
                    "threshold": float(th),
                    "auc": roc_auc,
                    "support": positives,
                }
            )

    # Micro
    fpr_micro, tpr_micro, thresholds_micro = roc_curve(y_true.ravel(), probs.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))
    for fp, tp, th in zip(fpr_micro, tpr_micro, thresholds_micro):
        records.append(
            {
                "curve_type": "micro",
                "class_name": "micro",
                "fpr": float(fp),
                "tpr": float(tp),
                "threshold": float(th),
                "auc": auc_micro,
                "support": int(len(labels)),
            }
        )

    # Macro + weighted curves via interpolation on a shared FPR grid.
    grid = np.linspace(0.0, 1.0, 1001)
    if per_class_curves:
        tprs = []
        weights = []
        per_class_auc_values = []
        for name, (fpr, tpr, roc_auc, support) in per_class_curves.items():
            tprs.append(np.interp(grid, fpr, tpr))
            weights.append(float(support))
            per_class_auc_values.append(float(roc_auc))

        mean_tpr = np.mean(np.vstack(tprs), axis=0)
        macro_curve_auc = float(auc(grid, mean_tpr))

        weights_np = np.array(weights, dtype=np.float64)
        weighted_tpr = np.average(np.vstack(tprs), axis=0, weights=weights_np)
        weighted_curve_auc = float(auc(grid, weighted_tpr))

        # Keep summary AUROCs aligned with metrics.py definitions.
        macro_auc = float(np.mean(per_class_auc_values))
        weighted_auc = float(np.sum(weights_np * np.array(per_class_auc_values, dtype=np.float64)) / np.sum(weights_np))
    else:
        mean_tpr = np.zeros_like(grid)
        weighted_tpr = np.zeros_like(grid)
        macro_curve_auc = float("nan")
        weighted_curve_auc = float("nan")
        macro_auc = float("nan")
        weighted_auc = float("nan")

    for fp, tp in zip(grid, mean_tpr):
        records.append(
            {
                "curve_type": "macro",
                "class_name": "macro",
                "fpr": float(fp),
                "tpr": float(tp),
                "threshold": float("nan"),
                "auc": macro_auc,
                "support": int(len(per_class_curves)),
            }
        )
    for fp, tp in zip(grid, weighted_tpr):
        records.append(
            {
                "curve_type": "weighted",
                "class_name": "weighted",
                "fpr": float(fp),
                "tpr": float(tp),
                "threshold": float("nan"),
                "auc": weighted_auc,
                "support": int(len(labels)),
            }
        )

    roc_df = pd.DataFrame.from_records(records)
    roc_points_path, format_used = _save_roc_points(roc_df, out_dir)

    # Plots
    _plot_overall(
        out_dir / "roc_overall.png",
        (fpr_micro, tpr_micro, auc_micro),
        (grid, mean_tpr, macro_auc),
        (grid, weighted_tpr, weighted_auc),
    )
    _plot_per_class(out_dir / "roc_per_class.png", per_class_curves)
    _plot_bar(out_dir / "roc_per_class_auroc_bar.png", per_class_curves)

    summary = {
        "micro_auroc": auc_micro,
        "macro_auroc": macro_auc,
        "weighted_auroc": weighted_auc,
        "macro_curve_auroc": macro_curve_auc,
        "weighted_curve_auroc": weighted_curve_auc,
        "per_class_auroc": {k: float(v[2]) for k, v in per_class_curves.items()},
        "per_class_support": {k: int(v[3]) for k, v in per_class_curves.items()},
        "roc_points_path": str(roc_points_path),
        "roc_points_format": format_used,
        "plot_paths": {
            "overall": str(out_dir / "roc_overall.png"),
            "per_class": str(out_dir / "roc_per_class.png"),
            "bar": str(out_dir / "roc_per_class_auroc_bar.png"),
        },
    }
    write_json(out_dir / "roc_summary.json", summary)
    return summary

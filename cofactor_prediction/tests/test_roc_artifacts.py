from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.roc import generate_roc_artifacts


def test_roc_artifacts_and_summary_consistency(tmp_path: Path) -> None:
    labels = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
    probs = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.05, 0.85, 0.10],
            [0.10, 0.20, 0.70],
            [0.20, 0.60, 0.20],
            [0.70, 0.15, 0.15],
            [0.10, 0.30, 0.60],
        ],
        dtype=np.float64,
    )
    class_names = ["no_cofactor", "cfA", "cfB"]

    out_dir = tmp_path / "roc"
    summary = generate_roc_artifacts(probs=probs, labels=labels, class_names=class_names, out_dir=out_dir)

    summary_path = out_dir / "roc_summary.json"
    assert summary_path.exists()

    loaded = json.loads(summary_path.read_text())
    assert loaded["roc_points_path"] == summary["roc_points_path"]

    points_path = Path(summary["roc_points_path"])
    assert points_path.exists()

    if points_path.suffix == ".parquet":
        points_df = pd.read_parquet(points_path)
    else:
        points_df = pd.read_csv(points_path)
    assert {"curve_type", "class_name", "fpr", "tpr", "threshold"}.issubset(set(points_df.columns))

    y_true = np.eye(3)[labels]
    expected_per_class = {
        class_names[i]: float(roc_auc_score(y_true[:, i], probs[:, i]))
        for i in range(3)
    }

    for name in class_names:
        assert abs(summary["per_class_auroc"][name] - expected_per_class[name]) < 1e-12

    macro_expected = float(np.mean(list(expected_per_class.values())))
    assert abs(summary["macro_auroc"] - macro_expected) < 1e-12

    assert (out_dir / "roc_overall.png").exists()
    assert (out_dir / "roc_per_class.png").exists()
    assert (out_dir / "roc_per_class_auroc_bar.png").exists()

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.metrics import compute_classification_metrics


def test_metrics_match_reference_aurocs() -> None:
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

    metrics = compute_classification_metrics(probs=probs, labels=labels, class_names=class_names)

    y_true = np.eye(3)[labels]
    expected_per_class = {
        class_names[i]: float(roc_auc_score(y_true[:, i], probs[:, i]))
        for i in range(3)
    }

    for name in class_names:
        assert abs(metrics["per_class_auroc"][name] - expected_per_class[name]) < 1e-12

    expected_macro = float(np.mean(list(expected_per_class.values())))
    expected_weighted = float(
        sum(expected_per_class[class_names[i]] * int((labels == i).sum()) for i in range(3)) / len(labels)
    )
    expected_micro = float(roc_auc_score(y_true.ravel(), probs.ravel()))

    assert abs(metrics["macro_auroc"] - expected_macro) < 1e-12
    assert abs(metrics["weighted_auroc"] - expected_weighted) < 1e-12
    assert abs(metrics["micro_auroc"] - expected_micro) < 1e-12

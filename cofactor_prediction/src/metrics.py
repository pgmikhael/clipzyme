from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.int32)
    y[np.arange(labels.shape[0]), labels.astype(int)] = 1
    return y


def compute_classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Dict:
    labels = labels.astype(int)
    pred_labels = np.argmax(probs, axis=1).astype(int)
    num_classes = probs.shape[1]

    accuracy = float(accuracy_score(labels, pred_labels))
    macro_f1 = float(f1_score(labels, pred_labels, average="macro", zero_division=0))
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        labels, pred_labels, average="macro", zero_division=0
    )

    support = {class_names[i]: int((labels == i).sum()) for i in range(num_classes)}

    y_true = _one_hot(labels, num_classes)

    per_class_auroc: Dict[str, float] = {}
    valid_mask = []
    for i, name in enumerate(class_names):
        positives = int(y_true[:, i].sum())
        negatives = int((1 - y_true[:, i]).sum())
        if positives > 0 and negatives > 0:
            auc = float(roc_auc_score(y_true[:, i], probs[:, i]))
            per_class_auroc[name] = auc
            valid_mask.append(i)

    if valid_mask:
        macro_auroc = float(np.mean([per_class_auroc[class_names[i]] for i in valid_mask]))
        weights = np.array([support[class_names[i]] for i in valid_mask], dtype=np.float64)
        auc_values = np.array([per_class_auroc[class_names[i]] for i in valid_mask], dtype=np.float64)
        weighted_auroc = float(np.sum(weights * auc_values) / max(np.sum(weights), 1.0))
    else:
        macro_auroc = float("nan")
        weighted_auroc = float("nan")

    try:
        micro_auroc = float(roc_auc_score(y_true.ravel(), probs.ravel()))
    except ValueError:
        micro_auroc = float("nan")

    conf = confusion_matrix(labels, pred_labels, labels=list(range(num_classes)))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "micro_auroc": micro_auroc,
        "macro_auroc": macro_auroc,
        "weighted_auroc": weighted_auroc,
        "per_class_auroc": per_class_auroc,
        "per_class_support": support,
        "confusion_matrix": conf.tolist(),
        "class_names": class_names,
    }


def save_confusion_matrix_csv(metrics: Dict, out_csv: Path) -> None:
    class_names = metrics["class_names"]
    conf = np.array(metrics["confusion_matrix"], dtype=np.int64)
    df = pd.DataFrame(conf, index=class_names, columns=class_names)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)


def macro_f1_from_predictions(labels: np.ndarray, pred_labels: np.ndarray) -> float:
    labels = labels.astype(int)
    pred_labels = pred_labels.astype(int)
    return float(f1_score(labels, pred_labels, average="macro", zero_division=0))

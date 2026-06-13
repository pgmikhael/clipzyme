from __future__ import annotations

from pathlib import Path

import numpy as np

from src.io_contracts import validate_prediction_npz
from src.metrics import compute_classification_metrics
from src.models_knn import run_knn_sweep
from src.models_mlp import MLPConfig, train_mlp
from src.roc import generate_roc_artifacts


def _toy_split(seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    def sample(n_per_class: int):
        xs = []
        ys = []
        for cls in range(3):
            noise = rng.normal(0.0, 0.25, size=(n_per_class, 5)).astype(np.float32)
            xs.append(centers[cls] + noise)
            ys.extend([cls] * n_per_class)
        return np.vstack(xs).astype(np.float32), np.array(ys, dtype=np.int64)

    train_x, train_y = sample(6)
    dev_x, dev_y = sample(3)
    test_x, test_y = sample(3)
    return train_x, train_y, dev_x, dev_y, test_x, test_y


def test_smoke_knn_mlp_eval_and_roc(tmp_path: Path) -> None:
    train_x, train_y, dev_x, dev_y, test_x, test_y = _toy_split(seed=7)
    class_names = ["no_cofactor", "cfA", "cfB"]

    # KNN smoke
    knn_probs = run_knn_sweep(train_x, train_y, test_x, k_list=[1], num_classes=3)[1]
    knn_preds = np.argmax(knn_probs, axis=1)

    knn_pred_path = tmp_path / "knn_predictions.npz"
    np.savez(
        knn_pred_path,
        probs=knn_probs,
        labels=test_y,
        pred_labels=knn_preds,
        protein_ids=np.array([f"t{i}" for i in range(len(test_y))], dtype=object),
        class_names=np.array(class_names, dtype=object),
    )
    validate_prediction_npz(knn_pred_path)

    metrics = compute_classification_metrics(knn_probs, test_y, class_names)
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    roc_summary = generate_roc_artifacts(knn_probs, test_y, class_names, out_dir=tmp_path / "knn_roc")
    assert "macro_auroc" in roc_summary

    # MLP smoke
    cfg = MLPConfig(
        name="mlp_smoke",
        hidden_dims=[16, 8],
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=8,
        max_epochs=8,
        patience=3,
        loss_type="ce",
        use_class_weights=False,
    )
    mlp_out = train_mlp(
        train_x=train_x,
        train_y=train_y,
        dev_x=dev_x,
        dev_y=dev_y,
        test_x=test_x,
        test_y=test_y,
        config=cfg,
        seed=42,
    )

    assert mlp_out["probs"].shape == (test_x.shape[0], 3)
    assert mlp_out["labels"].shape[0] == test_x.shape[0]
    assert len(mlp_out["history"]) >= 1

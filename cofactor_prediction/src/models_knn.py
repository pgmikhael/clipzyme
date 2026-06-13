from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)


def knn_weighted_probabilities(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    num_classes: int,
    k: int,
    batch_size: int = 512,
) -> np.ndarray:
    train_x = _l2_normalize(train_x.astype(np.float32))
    test_x = _l2_normalize(test_x.astype(np.float32))
    train_y = train_y.astype(np.int64)

    n_test = test_x.shape[0]
    k = min(k, train_x.shape[0])
    probs = np.zeros((n_test, num_classes), dtype=np.float32)

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        sims = test_x[start:end] @ train_x.T
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_sim = np.take_along_axis(sims, top_idx, axis=1)

        # Sort top-k by similarity descending for stable behavior.
        order = np.argsort(-top_sim, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sim = np.take_along_axis(top_sim, order, axis=1)

        scores = np.zeros((end - start, num_classes), dtype=np.float32)
        for rank in range(k):
            cls = train_y[top_idx[:, rank]]
            weights = top_sim[:, rank]
            scores[np.arange(end - start), cls] += weights

        probs[start:end] = _softmax(scores)

    return probs


def run_knn_sweep(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k_list: Iterable[int],
    num_classes: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for k in k_list:
        out[int(k)] = knn_weighted_probabilities(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            num_classes=num_classes,
            k=int(k),
        )
    return out

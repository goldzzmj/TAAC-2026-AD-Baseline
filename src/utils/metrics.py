from __future__ import annotations

from typing import Any

import numpy as np


def binary_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    positives = labels == 1
    negatives = labels == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[positives].sum()
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    binary_labels: np.ndarray,
) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    accuracy = float((predictions == labels).mean())

    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    positive_scores = probabilities[:, -1]
    auc = binary_auc_score(binary_labels, positive_scores)

    return {
        "accuracy": accuracy,
        "auc": auc,
    }

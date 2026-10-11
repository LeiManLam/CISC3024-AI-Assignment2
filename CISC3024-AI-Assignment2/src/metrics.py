"""Evaluation metrics for probabilistic classifiers."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def negative_log_likelihood(
    y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-12
) -> float:
    """Mean NLL = -mean log p(y|x) using predictive probabilities."""
    n = len(y_true)
    p = probs[np.arange(n), y_true]
    return float(-np.log(p + eps).mean())


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).
    Confidence = max predicted class probability.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

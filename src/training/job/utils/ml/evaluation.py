"""Evaluation helpers for imbalanced binary classifiers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_binary_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_probability: Sequence[float] | np.ndarray,
    top_fractions: Sequence[float] = (0.001, 0.005, 0.01),
) -> dict[str, Any]:
    """Evaluate positive-class probabilities for an imbalanced target.

    Args:
        y_true: Binary ground-truth labels encoded as zero and one.
        y_probability: Predicted probabilities for positive class one.
        top_fractions: Fractions of highest-risk observations to inspect.

    Returns:
        Overall discrimination and calibration metrics, plus precision,
        recall, and lift within each requested highest-risk fraction.
    """
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_probability, dtype=float)

    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError(
            "Labels and probabilities must have the same non-zero length."
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Ground-truth labels must contain only zero and one.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation data must contain both binary classes.")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Predictions must be finite probabilities in [0, 1].")

    prevalence = float(labels.mean())
    positives = int(labels.sum())
    ranked_indices = np.argsort(-probabilities, kind="stable")
    top_fraction_metrics = []

    for fraction in top_fractions:
        if not 0 < fraction <= 1:
            raise ValueError(
                "Every top fraction must be in the interval (0, 1]."
            )

        selected_count = max(1, int(np.ceil(len(labels) * fraction)))
        selected_indices = ranked_indices[:selected_count]
        selected_positives = int(labels[selected_indices].sum())
        precision = selected_positives / selected_count

        top_fraction_metrics.append(
            {
                "fraction": float(fraction),
                "selected": selected_count,
                "positives": selected_positives,
                "precision": float(precision),
                "recall": float(selected_positives / positives),
                "lift": float(precision / prevalence),
            }
        )

    return {
        "observations": len(labels),
        "positives": positives,
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "top_fraction_metrics": top_fraction_metrics,
    }

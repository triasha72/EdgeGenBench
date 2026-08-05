"""Classification metrics for aircraft-design feasibility."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_binary_inputs(
    actual: Sequence[bool | int],
    probability: Sequence[float],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary targets and feasibility probabilities."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one.")

    actual_values = np.asarray(
        actual,
        dtype=np.int64,
    )
    probability_values = np.asarray(
        probability,
        dtype=np.float64,
    )

    if actual_values.ndim != 1:
        raise ValueError("Actual class values must be one-dimensional.")

    if probability_values.ndim != 1:
        raise ValueError("Probability values must be one-dimensional.")

    if len(actual_values) == 0:
        raise ValueError("At least one observation is required.")

    if len(actual_values) != len(probability_values):
        raise ValueError("Actual classes and probabilities must have identical lengths.")

    unique_classes = set(np.unique(actual_values).tolist())

    if not unique_classes.issubset({0, 1}):
        raise ValueError("Actual class values must be binary.")

    if not np.isfinite(probability_values).all():
        raise ValueError("Probability values must be finite.")

    if np.any((probability_values < 0.0) | (probability_values > 1.0)):
        raise ValueError("Probability values must be between zero and one.")

    return actual_values, probability_values


def calculate_classification_metrics(
    actual: Sequence[bool | int],
    probability: Sequence[float],
    threshold: float = 0.50,
) -> dict[str, float | int]:
    """Calculate feasibility classification metrics.

    A false-safe prediction occurs when an actually infeasible
    design is predicted to be feasible.
    """
    actual_values, probability_values = _validate_binary_inputs(
        actual=actual,
        probability=probability,
        threshold=threshold,
    )

    predicted_values = (probability_values >= threshold).astype(np.int64)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual_values,
        predicted_values,
        labels=[0, 1],
    ).ravel()

    infeasible_count = true_negative + false_positive
    feasible_count = true_positive + false_negative

    if infeasible_count > 0:
        false_safe_rate = false_positive / infeasible_count
    else:
        false_safe_rate = float("nan")

    if feasible_count > 0:
        false_reject_rate = false_negative / feasible_count
    else:
        false_reject_rate = float("nan")

    if len(np.unique(actual_values)) == 2:
        roc_auc = float(
            roc_auc_score(
                actual_values,
                probability_values,
            )
        )
        pr_auc = float(
            average_precision_score(
                actual_values,
                probability_values,
            )
        )
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                actual_values,
                predicted_values,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual_values,
                predicted_values,
            )
        ),
        "precision": float(
            precision_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(
            brier_score_loss(
                actual_values,
                probability_values,
            )
        ),
        "false_safe_rate": float(false_safe_rate),
        "false_reject_rate": float(false_reject_rate),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "sample_count": int(len(actual_values)),
        "feasible_count": int(feasible_count),
        "infeasible_count": int(infeasible_count),
    }


def build_confusion_matrix(
    actual: Sequence[bool | int],
    probability: Sequence[float],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Create a labeled two-by-two confusion matrix."""
    actual_values, probability_values = _validate_binary_inputs(
        actual=actual,
        probability=probability,
        threshold=threshold,
    )

    predicted_values = (probability_values >= threshold).astype(np.int64)

    matrix = confusion_matrix(
        actual_values,
        predicted_values,
        labels=[0, 1],
    )

    return pd.DataFrame(
        matrix,
        index=pd.Index(
            [
                "actual_infeasible",
                "actual_feasible",
            ],
            name="actual_class",
        ),
        columns=pd.Index(
            [
                "predicted_infeasible",
                "predicted_feasible",
            ],
            name="predicted_class",
        ),
    )


def build_probability_calibration(
    actual: Sequence[bool | int],
    probability: Sequence[float],
    bin_count: int = 10,
) -> pd.DataFrame:
    """Summarize observed feasibility by probability bin."""
    if bin_count < 1:
        raise ValueError("bin_count must be at least one.")

    actual_values, probability_values = _validate_binary_inputs(
        actual=actual,
        probability=probability,
        threshold=0.50,
    )

    bin_indices = np.minimum(
        (probability_values * float(bin_count)).astype(np.int64),
        bin_count - 1,
    )

    records: list[dict[str, float | int]] = []

    for bin_index in range(bin_count):
        mask = bin_indices == bin_index

        if not np.any(mask):
            continue

        lower_bound = bin_index / float(bin_count)
        upper_bound = (bin_index + 1) / float(bin_count)

        mean_probability = float(np.mean(probability_values[mask]))
        empirical_feasible_rate = float(np.mean(actual_values[mask]))

        records.append(
            {
                "probability_bin": (bin_index + 1),
                "bin_lower": lower_bound,
                "bin_upper": upper_bound,
                "sample_count": int(np.sum(mask)),
                "mean_probability": (mean_probability),
                "empirical_feasible_rate": (empirical_feasible_rate),
                "calibration_error": (empirical_feasible_rate - mean_probability),
            }
        )

    return pd.DataFrame(records)

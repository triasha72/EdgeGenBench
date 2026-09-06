"""Validation-only selection for real-flight anomaly candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, recall_score


def selective_metrics(
    expected: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Measure quality and review burden when low-confidence rows abstain."""
    expected = np.asarray(expected)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(expected) != len(probabilities):
        raise ValueError("probabilities must have one row per expected label")
    confidence = probabilities.max(axis=1)
    covered = confidence >= threshold
    prediction = probabilities.argmax(axis=1)
    if not np.any(covered):
        return {"coverage": 0.0, "macro_f1": 0.0, "minimum_anomaly_recall": 0.0}
    recalls = recall_score(
        expected[covered], prediction[covered], labels=[1, 2, 3],
        average=None, zero_division=0,
    )
    return {
        "coverage": float(np.mean(covered)),
        "macro_f1": float(f1_score(
            expected[covered], prediction[covered], labels=[0, 1, 2, 3],
            average="macro", zero_division=0,
        )),
        "minimum_anomaly_recall": float(np.min(recalls)),
    }


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose only a validation candidate meeting frozen quality and coverage gates."""
    eligible = [
        record for record in records
        if record["validation"]["macro_f1"] >= 0.75
        and record["validation"]["minimum_anomaly_recall"] >= 0.60
        and record["validation"]["coverage"] >= 0.80
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["validation"]["minimum_anomaly_recall"],
            row["validation"]["macro_f1"],
            row["validation"]["coverage"],
        ),
    )

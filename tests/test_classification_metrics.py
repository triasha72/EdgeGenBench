"""Tests for feasibility classification metrics."""

import numpy as np
import pytest

from edgegenbench.evaluation.classification import (
    build_confusion_matrix,
    build_probability_calibration,
    calculate_classification_metrics,
)


def test_classification_metrics_include_false_safe_rate() -> None:
    """False-safe predictions should be reported explicitly."""
    actual = [0, 0, 1, 1]
    probability = [0.10, 0.80, 0.60, 0.90]

    metrics = calculate_classification_metrics(
        actual=actual,
        probability=probability,
        threshold=0.50,
    )

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["true_positive"] == 2

    assert metrics["false_safe_rate"] == pytest.approx(0.50)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(2.0 / 3.0)


def test_confusion_matrix_has_clear_labels() -> None:
    """The confusion matrix should identify unsafe acceptance."""
    matrix = build_confusion_matrix(
        actual=[0, 0, 1, 1],
        probability=[0.10, 0.80, 0.60, 0.90],
        threshold=0.50,
    )

    assert (
        matrix.loc[
            "actual_infeasible",
            "predicted_feasible",
        ]
        == 1
    )

    assert (
        matrix.loc[
            "actual_feasible",
            "predicted_feasible",
        ]
        == 2
    )


def test_probability_calibration_counts_all_rows() -> None:
    """Calibration bins should retain every observation."""
    calibration = build_probability_calibration(
        actual=[0, 0, 1, 1, 1],
        probability=[
            0.05,
            0.25,
            0.55,
            0.75,
            0.95,
        ],
        bin_count=5,
    )

    assert calibration["sample_count"].sum() == 5

    assert calibration["mean_probability"].between(0.0, 1.0).all()

    assert calibration["empirical_feasible_rate"].between(0.0, 1.0).all()


def test_invalid_probability_is_rejected() -> None:
    """Probabilities outside zero to one should fail."""
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        calculate_classification_metrics(
            actual=[0, 1],
            probability=[0.20, 1.20],
        )


def test_single_class_auc_is_nan() -> None:
    """Undefined ranking metrics should be represented by NaN."""
    metrics = calculate_classification_metrics(
        actual=[1, 1, 1],
        probability=[0.60, 0.70, 0.80],
    )

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])

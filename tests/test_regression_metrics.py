"""Tests for regression evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from edgegenbench.evaluation.regression import (
    calculate_regression_metrics,
)


def test_perfect_predictions_have_zero_error() -> None:
    actual = pd.DataFrame(
        {
            "target_a": [1.0, 2.0, 3.0],
            "target_b": [10.0, 20.0, 30.0],
        }
    )
    predicted = actual.copy()

    metrics = calculate_regression_metrics(
        actual=actual,
        predicted=predicted,
        targets=("target_a", "target_b"),
    )

    np.testing.assert_allclose(metrics["mae"], 0.0)
    np.testing.assert_allclose(metrics["rmse"], 0.0)
    np.testing.assert_allclose(metrics["nrmse_std"], 0.0)
    np.testing.assert_allclose(metrics["r2"], 1.0)


def test_missing_prediction_target_raises_error() -> None:
    actual = pd.DataFrame(
        {
            "target_a": [1.0, 2.0],
            "target_b": [3.0, 4.0],
        }
    )
    predicted = pd.DataFrame({"target_a": [1.0, 2.0]})

    with pytest.raises(
        ValueError,
        match="Predictions are missing targets",
    ):
        calculate_regression_metrics(
            actual=actual,
            predicted=predicted,
            targets=("target_a", "target_b"),
        )

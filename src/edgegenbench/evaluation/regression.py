"""Regression metrics for surrogate-model evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def calculate_regression_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Calculate target-level regression metrics."""
    target_names = tuple(targets)

    missing_actual = sorted(set(target_names).difference(actual.columns))
    missing_predicted = sorted(set(target_names).difference(predicted.columns))

    if missing_actual:
        raise ValueError(f"Actual values are missing targets: {missing_actual}")

    if missing_predicted:
        raise ValueError(f"Predictions are missing targets: {missing_predicted}")

    actual_values = actual.loc[:, list(target_names)].to_numpy(dtype=np.float64)
    predicted_values = predicted.loc[:, list(target_names)].to_numpy(dtype=np.float64)

    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted arrays must have identical shapes.")

    if len(actual_values) == 0:
        raise ValueError("At least one observation is required.")

    errors = predicted_values - actual_values

    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))

    target_standard_deviation = np.std(actual_values, axis=0)
    nrmse_standard_deviation = np.divide(
        rmse,
        target_standard_deviation,
        out=np.full_like(rmse, np.nan),
        where=target_standard_deviation > np.finfo(np.float64).eps,
    )

    residual_sum_of_squares = np.sum(np.square(errors), axis=0)
    centered_actual = actual_values - np.mean(actual_values, axis=0)
    total_sum_of_squares = np.sum(np.square(centered_actual), axis=0)

    r2 = np.divide(
        residual_sum_of_squares,
        total_sum_of_squares,
        out=np.full_like(residual_sum_of_squares, np.nan),
        where=total_sum_of_squares > np.finfo(np.float64).eps,
    )
    r2 = 1.0 - r2

    return pd.DataFrame(
        {
            "target": target_names,
            "mae": mae,
            "rmse": rmse,
            "nrmse_std": nrmse_standard_deviation,
            "r2": r2,
        }
    )

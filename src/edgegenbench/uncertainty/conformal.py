"""Split-conformal prediction intervals for EdgeGenBench."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_coverage(coverage: float) -> None:
    """Validate a requested interval coverage."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one.")


def _require_targets(
    frame: pd.DataFrame,
    targets: Sequence[str],
    frame_name: str,
) -> None:
    """Validate required target columns."""
    missing_targets = sorted(set(targets).difference(frame.columns))

    if missing_targets:
        raise ValueError(f"{frame_name} is missing targets: {missing_targets}")


def calculate_conformal_quantiles(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    targets: Sequence[str],
    coverage: float = 0.90,
) -> pd.Series:
    """Calculate finite-sample split-conformal residual quantiles."""
    _validate_coverage(coverage)

    target_names = tuple(targets)

    if not target_names:
        raise ValueError("At least one target must be supplied.")

    _require_targets(
        frame=actual,
        targets=target_names,
        frame_name="Actual values",
    )
    _require_targets(
        frame=predicted,
        targets=target_names,
        frame_name="Predictions",
    )

    actual_values = actual.loc[
        :,
        list(target_names),
    ].to_numpy(dtype=np.float64)

    predicted_values = predicted.loc[
        :,
        list(target_names),
    ].to_numpy(dtype=np.float64)

    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have identical shapes.")

    sample_count = len(actual_values)

    if sample_count == 0:
        raise ValueError("At least one calibration observation is required.")

    absolute_residuals = np.abs(actual_values - predicted_values)

    quantile_level = min(
        1.0,
        np.ceil((sample_count + 1) * coverage) / sample_count,
    )

    quantiles = np.quantile(
        absolute_residuals,
        quantile_level,
        axis=0,
        method="higher",
    )

    return pd.Series(
        quantiles,
        index=target_names,
        name="conformal_quantile",
        dtype=np.float64,
    )


def build_conformal_intervals(
    predictions: pd.DataFrame,
    quantiles: pd.Series,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Apply calibrated residual quantiles to point predictions."""
    target_names = tuple(targets)

    _require_targets(
        frame=predictions,
        targets=target_names,
        frame_name="Predictions",
    )

    missing_quantiles = sorted(set(target_names).difference(quantiles.index))

    if missing_quantiles:
        raise ValueError(f"Conformal quantiles are missing targets: {missing_quantiles}")

    output = pd.DataFrame(index=predictions.index)

    for target in target_names:
        prediction_values = predictions[target].to_numpy(dtype=np.float64)
        interval_radius = float(quantiles[target])

        if interval_radius < 0.0:
            raise ValueError("Conformal quantiles cannot be negative.")

        output[f"prediction_{target}"] = prediction_values
        output[f"lower_{target}"] = prediction_values - interval_radius
        output[f"upper_{target}"] = prediction_values + interval_radius

    return output

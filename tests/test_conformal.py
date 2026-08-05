"""Tests for split-conformal prediction intervals."""

import numpy as np
import pandas as pd
import pytest

from edgegenbench.uncertainty.conformal import (
    build_conformal_intervals,
    calculate_conformal_quantiles,
)


def _calibration_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = pd.DataFrame(
        {
            "mass": [
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
            ],
            "energy": [
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
            ],
        }
    )

    predicted = pd.DataFrame(
        {
            "mass": [
                98.0,
                111.0,
                117.0,
                134.0,
                139.0,
            ],
            "energy": [
                39.0,
                47.0,
                49.0,
                52.0,
                63.0,
            ],
        }
    )

    return actual, predicted


def test_conformal_quantiles_are_nonnegative() -> None:
    """Calibration should produce one nonnegative value per target."""
    actual, predicted = _calibration_data()

    quantiles = calculate_conformal_quantiles(
        actual=actual,
        predicted=predicted,
        targets=("mass", "energy"),
        coverage=0.80,
    )

    assert tuple(quantiles.index) == (
        "mass",
        "energy",
    )
    assert np.isfinite(quantiles.to_numpy()).all()
    assert (quantiles >= 0.0).all()


def test_conformal_intervals_surround_predictions() -> None:
    """Each interval should contain its point prediction."""
    actual, predicted = _calibration_data()

    quantiles = calculate_conformal_quantiles(
        actual=actual,
        predicted=predicted,
        targets=("mass", "energy"),
        coverage=0.80,
    )

    test_predictions = pd.DataFrame(
        {
            "mass": [105.0, 125.0],
            "energy": [42.0, 57.0],
        }
    )

    intervals = build_conformal_intervals(
        predictions=test_predictions,
        quantiles=quantiles,
        targets=("mass", "energy"),
    )

    for target in ("mass", "energy"):
        point = intervals[f"prediction_{target}"]
        lower = intervals[f"lower_{target}"]
        upper = intervals[f"upper_{target}"]

        assert (lower <= point).all()
        assert (point <= upper).all()


def test_invalid_conformal_coverage_is_rejected() -> None:
    """Invalid requested coverage should fail clearly."""
    actual, predicted = _calibration_data()

    with pytest.raises(
        ValueError,
        match="coverage must be between",
    ):
        calculate_conformal_quantiles(
            actual=actual,
            predicted=predicted,
            targets=("mass", "energy"),
            coverage=0.0,
        )


def test_missing_target_is_rejected() -> None:
    """Missing prediction targets should produce a clear error."""
    actual, predicted = _calibration_data()

    invalid_predictions = predicted.drop(columns=["energy"])

    with pytest.raises(
        ValueError,
        match="Predictions is missing targets",
    ):
        calculate_conformal_quantiles(
            actual=actual,
            predicted=invalid_predictions,
            targets=("mass", "energy"),
        )

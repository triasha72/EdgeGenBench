"""Tests for Random-Forest ensemble uncertainty."""

import numpy as np
import pandas as pd
import pytest

from edgegenbench.models.tree_surrogate import (
    HIST_GRADIENT_BOOSTING,
    RANDOM_FOREST,
    TreeSurrogate,
)
from edgegenbench.uncertainty.ensemble import (
    predict_tree_ensemble_intervals,
)


def _make_training_frame(
    sample_count: int = 140,
) -> pd.DataFrame:
    """Create a deterministic synthetic regression dataset."""
    random_generator = np.random.default_rng(42)

    architectures = np.resize(
        np.asarray(
            [
                "conventional_turboprop",
                "parallel_hybrid",
                "series_hybrid",
                "fuel_cell_electric",
            ]
        ),
        sample_count,
    )

    architecture_effect = (
        pd.Series(architectures)
        .map(
            {
                "conventional_turboprop": 0.0,
                "parallel_hybrid": 200.0,
                "series_hybrid": 400.0,
                "fuel_cell_electric": 600.0,
            }
        )
        .to_numpy()
    )

    frame = pd.DataFrame(
        {
            "passenger_capacity": random_generator.integers(
                40,
                91,
                size=sample_count,
            ),
            "design_range_km": random_generator.uniform(
                400.0,
                1500.0,
                size=sample_count,
            ),
            "cruise_speed_kmh": random_generator.uniform(
                420.0,
                650.0,
                size=sample_count,
            ),
            "battery_specific_energy_wh_per_kg": (
                random_generator.uniform(
                    300.0,
                    750.0,
                    size=sample_count,
                )
            ),
            "hydrogen_storage_efficiency": (
                random_generator.uniform(
                    0.45,
                    0.70,
                    size=sample_count,
                )
            ),
            "hybridization_ratio": random_generator.uniform(
                0.0,
                0.65,
                size=sample_count,
            ),
            "propulsion_architecture": architectures,
        }
    )

    frame["target_mass"] = (
        8000.0
        + 85.0 * frame["passenger_capacity"]
        + 1.4 * frame["design_range_km"]
        + architecture_effect
    )

    frame["target_energy"] = (
        500.0
        + 7.0 * frame["design_range_km"]
        + 2.5 * frame["cruise_speed_kmh"]
        - 350.0 * frame["hybridization_ratio"]
        + 0.25 * architecture_effect
    )

    return frame


def test_random_forest_intervals_are_created() -> None:
    """Random Forest should produce valid empirical intervals."""
    frame = _make_training_frame()

    model = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=("target_mass", "target_energy"),
        parameters={
            "n_estimators": 20,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
    )

    intervals = predict_tree_ensemble_intervals(
        model=model,
        frame=frame.iloc[:12],
        coverage=0.90,
    )

    assert len(intervals) == 12

    for target in ("target_mass", "target_energy"):
        prediction = intervals[f"prediction_{target}"]
        uncertainty = intervals[f"uncertainty_std_{target}"]
        lower = intervals[f"lower_{target}"]
        upper = intervals[f"upper_{target}"]

        assert np.isfinite(prediction).all()
        assert np.isfinite(uncertainty).all()
        assert np.isfinite(lower).all()
        assert np.isfinite(upper).all()

        assert (uncertainty >= 0.0).all()
        assert (lower <= prediction).all()
        assert (prediction <= upper).all()


def test_invalid_coverage_is_rejected() -> None:
    """Coverage must be strictly between zero and one."""
    frame = _make_training_frame()

    model = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=("target_mass", "target_energy"),
        parameters={"n_estimators": 10},
    )

    with pytest.raises(
        ValueError,
        match="coverage must be between",
    ):
        predict_tree_ensemble_intervals(
            model=model,
            frame=frame.iloc[:5],
            coverage=1.0,
        )


def test_non_random_forest_model_is_rejected() -> None:
    """Ensemble intervals should reject unsupported model families."""
    frame = _make_training_frame()

    model = TreeSurrogate.fit(
        frame=frame,
        model_type=HIST_GRADIENT_BOOSTING,
        targets=("target_mass", "target_energy"),
        parameters={
            "max_iter": 10,
            "max_leaf_nodes": 7,
        },
    )

    with pytest.raises(
        ValueError,
        match="requires a Random Forest",
    ):
        predict_tree_ensemble_intervals(
            model=model,
            frame=frame.iloc[:5],
        )

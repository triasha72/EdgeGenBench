"""Tests for the feasibility classifier."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegenbench.models.feasibility import (
    FeasibilityClassifier,
)


def _make_classification_frame(
    sample_count: int = 180,
) -> pd.DataFrame:
    """Create deterministic aircraft-design classification data."""
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

    frame = pd.DataFrame(
        {
            "passenger_capacity": (
                random_generator.integers(
                    40,
                    91,
                    size=sample_count,
                )
            ),
            "design_range_km": (
                random_generator.uniform(
                    400.0,
                    1500.0,
                    size=sample_count,
                )
            ),
            "cruise_speed_kmh": (
                random_generator.uniform(
                    420.0,
                    650.0,
                    size=sample_count,
                )
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
            "hybridization_ratio": (
                random_generator.uniform(
                    0.0,
                    0.65,
                    size=sample_count,
                )
            ),
            "propulsion_architecture": architectures,
        }
    )

    frame["is_feasible"] = (
        (frame["design_range_km"] < 1050.0)
        & (frame["hybridization_ratio"] < 0.45)
        & (frame["battery_specific_energy_wh_per_kg"] > 390.0)
    )

    return frame


def test_classifier_predicts_probabilities() -> None:
    """Classifier probabilities should be finite and bounded."""
    frame = _make_classification_frame()

    model = FeasibilityClassifier.fit(
        frame=frame,
        parameters={
            "n_estimators": 25,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
    )

    probabilities = model.predict_feasibility_probability(frame.iloc[:12])

    predictions = model.predict(frame.iloc[:12])

    assert len(probabilities) == 12
    assert len(predictions) == 12

    assert np.isfinite(probabilities.to_numpy()).all()

    assert probabilities.between(
        0.0,
        1.0,
    ).all()

    assert predictions.dtype == bool


def test_classifier_save_load_round_trip(
    tmp_path: Path,
) -> None:
    """Serialized classifiers should preserve probabilities."""
    frame = _make_classification_frame()

    model = FeasibilityClassifier.fit(
        frame=frame,
        parameters={
            "n_estimators": 20,
            "max_depth": 7,
        },
        threshold=0.70,
    )

    expected_probabilities = model.predict_feasibility_probability(frame.iloc[:10])

    model_path = tmp_path / "feasibility_model.joblib"
    model.save(model_path)

    loaded_model = FeasibilityClassifier.load(model_path)

    loaded_probabilities = loaded_model.predict_feasibility_probability(frame.iloc[:10])

    assert loaded_model.threshold == pytest.approx(0.70)

    np.testing.assert_allclose(
        loaded_probabilities.to_numpy(),
        expected_probabilities.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_threshold_changes_acceptance() -> None:
    """Higher thresholds should never accept more designs."""
    frame = _make_classification_frame()

    model = FeasibilityClassifier.fit(
        frame=frame,
        parameters={"n_estimators": 20},
    )

    low_threshold_predictions = model.predict(
        frame.iloc[:30],
        threshold=0.30,
    )

    high_threshold_predictions = model.predict(
        frame.iloc[:30],
        threshold=0.80,
    )

    assert high_threshold_predictions.sum() <= low_threshold_predictions.sum()


def test_unknown_architecture_is_rejected() -> None:
    """Prediction should reject unseen architecture labels."""
    frame = _make_classification_frame()

    model = FeasibilityClassifier.fit(
        frame=frame,
        parameters={"n_estimators": 15},
    )

    invalid_frame = frame.iloc[:1].copy()
    invalid_frame["propulsion_architecture"] = "unknown_architecture"

    with pytest.raises(
        ValueError,
        match="unknown categories",
    ):
        model.predict_feasibility_probability(invalid_frame)


def test_single_class_training_is_rejected() -> None:
    """Training requires both feasible and infeasible cases."""
    frame = _make_classification_frame()
    frame["is_feasible"] = True

    with pytest.raises(
        ValueError,
        match="both feasible and infeasible",
    ):
        FeasibilityClassifier.fit(frame)

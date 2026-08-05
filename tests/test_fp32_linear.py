"""Tests for the FP32 linear surrogate."""

from pathlib import Path

import numpy as np
import pandas as pd

from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    NUMERIC_FEATURES,
    FP32LinearSurrogate,
)


def _make_training_frame() -> pd.DataFrame:
    random_generator = np.random.default_rng(42)
    sample_count = 80

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
            "hydrogen_storage_efficiency": random_generator.uniform(
                0.45,
                0.70,
                size=sample_count,
            ),
            "hybridization_ratio": random_generator.uniform(
                0.0,
                0.65,
                size=sample_count,
            ),
            CATEGORICAL_FEATURE: np.resize(
                np.asarray(
                    [
                        "conventional_turboprop",
                        "parallel_hybrid",
                        "series_hybrid",
                        "fuel_cell_electric",
                    ]
                ),
                sample_count,
            ),
        }
    )

    architecture_effect = frame[CATEGORICAL_FEATURE].map(
        {
            "conventional_turboprop": 0.0,
            "parallel_hybrid": 200.0,
            "series_hybrid": 400.0,
            "fuel_cell_electric": 600.0,
        }
    )

    frame["target_mass"] = (
        5000.0
        + 80.0 * frame["passenger_capacity"]
        + 1.5 * frame["design_range_km"]
        + architecture_effect
    )

    frame["target_energy"] = (
        1000.0
        + 4.0 * frame["design_range_km"]
        + 2.0 * frame["cruise_speed_kmh"]
        - 250.0 * frame["hybridization_ratio"]
    )

    return frame


def test_fit_predict_and_save_round_trip(tmp_path: Path) -> None:
    frame = _make_training_frame()
    targets = ("target_mass", "target_energy")

    model = FP32LinearSurrogate.fit(
        frame=frame,
        targets=targets,
        alpha=1.0e-4,
    )

    predictions = model.predict(frame)

    assert tuple(predictions.columns) == targets
    assert predictions.shape == (len(frame), len(targets))
    assert predictions.to_numpy().dtype == np.float32

    maximum_error = np.abs(predictions.to_numpy() - frame.loc[:, targets].to_numpy()).max()

    assert maximum_error < 1.0

    model_path = tmp_path / "fp32_linear_model.npz"
    model.save(model_path)

    loaded_model = FP32LinearSurrogate.load(model_path)
    loaded_predictions = loaded_model.predict(frame)

    np.testing.assert_allclose(
        loaded_predictions.to_numpy(),
        predictions.to_numpy(),
        rtol=1.0e-6,
        atol=1.0e-4,
    )


def test_model_rejects_unknown_architecture() -> None:
    frame = _make_training_frame()
    targets = ("target_mass", "target_energy")

    model = FP32LinearSurrogate.fit(
        frame=frame,
        targets=targets,
    )

    invalid_frame = frame.iloc[:1].copy()
    invalid_frame[CATEGORICAL_FEATURE] = "unknown_architecture"

    try:
        model.predict(invalid_frame)
    except ValueError as error:
        assert "Unknown propulsion_architecture" in str(error)
    else:
        raise AssertionError("Expected an unknown-architecture error.")


def test_feature_definition_is_stable() -> None:
    assert NUMERIC_FEATURES == (
        "passenger_capacity",
        "design_range_km",
        "cruise_speed_kmh",
        "battery_specific_energy_wh_per_kg",
        "hydrogen_storage_efficiency",
        "hybridization_ratio",
    )

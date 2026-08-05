"""Tests for nonlinear EdgeGenBench surrogate models."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegenbench.models.tree_surrogate import (
    HIST_GRADIENT_BOOSTING,
    RANDOM_FOREST,
    TreeSurrogate,
)


def _make_training_frame(sample_count: int = 120) -> pd.DataFrame:
    """Create a small deterministic multi-output regression dataset."""
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


@pytest.mark.parametrize(
    ("model_type", "parameters"),
    [
        (
            RANDOM_FOREST,
            {
                "n_estimators": 12,
                "max_depth": 8,
                "min_samples_leaf": 1,
            },
        ),
        (
            HIST_GRADIENT_BOOSTING,
            {
                "max_iter": 25,
                "max_leaf_nodes": 15,
                "learning_rate": 0.1,
            },
        ),
    ],
)
def test_tree_models_fit_and_predict(
    model_type: str,
    parameters: dict[str, int | float],
) -> None:
    """Both nonlinear model types should fit and return valid predictions."""
    frame = _make_training_frame()

    model = TreeSurrogate.fit(
        frame=frame,
        model_type=model_type,
        targets=("target_mass", "target_energy"),
        parameters=parameters,
    )

    predictions = model.predict(frame.iloc[:10])

    assert predictions.shape == (10, 2)
    assert tuple(predictions.columns) == (
        "target_mass",
        "target_energy",
    )
    assert np.isfinite(predictions.to_numpy()).all()


def test_random_forest_save_load_round_trip(
    tmp_path: Path,
) -> None:
    """A serialized model should reproduce the original predictions."""
    frame = _make_training_frame()

    model = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=("target_mass", "target_energy"),
        parameters={
            "n_estimators": 10,
            "max_depth": 6,
        },
    )

    expected_predictions = model.predict(frame.iloc[:8])

    model_path = tmp_path / "tree_model.joblib"
    model.save(model_path)

    assert model_path.exists()

    loaded_model = TreeSurrogate.load(model_path)
    loaded_predictions = loaded_model.predict(frame.iloc[:8])

    np.testing.assert_allclose(
        loaded_predictions.to_numpy(),
        expected_predictions.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_unknown_architecture_is_rejected() -> None:
    """The preprocessing pipeline should reject unseen categories."""
    frame = _make_training_frame()

    model = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=("target_mass", "target_energy"),
        parameters={"n_estimators": 10},
    )

    invalid_frame = frame.iloc[:1].copy()
    invalid_frame["propulsion_architecture"] = "unknown_architecture"

    with pytest.raises(ValueError, match="unknown categories"):
        model.predict(invalid_frame)


def test_missing_feature_column_is_rejected() -> None:
    """Prediction should fail clearly when a required input is absent."""
    frame = _make_training_frame()

    model = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=("target_mass", "target_energy"),
        parameters={"n_estimators": 10},
    )

    invalid_frame = frame.iloc[:2].drop(columns=["design_range_km"])

    with pytest.raises(
        ValueError,
        match="missing required feature columns",
    ):
        model.predict(invalid_frame)


def test_unsupported_model_type_is_rejected() -> None:
    """Unknown model names should produce a clear error."""
    frame = _make_training_frame()

    with pytest.raises(ValueError, match="Unsupported model type"):
        TreeSurrogate.fit(
            frame=frame,
            model_type="unsupported_model",
            targets=("target_mass", "target_energy"),
        )

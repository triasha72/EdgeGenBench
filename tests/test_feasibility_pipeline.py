"""Integration tests for feasibility-classifier training."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegenbench.training.feasibility import (
    select_feasibility_threshold,
    train_feasibility_classifier,
)


def _create_dataset(
    path: Path,
) -> None:
    """Create deterministic classification data with splits."""
    random_generator = np.random.default_rng(42)

    sample_count = 300

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
            "propulsion_architecture": (architectures),
        }
    )

    frame["is_feasible"] = (
        (frame["design_range_km"] < 1050.0)
        & (frame["hybridization_ratio"] < 0.45)
        & (frame["battery_specific_energy_wh_per_kg"] > 390.0)
    )

    frame["split"] = ["train"] * 180 + ["validation"] * 60 + ["test"] * 60

    frame.to_csv(path, index=False)


def test_threshold_selection_prioritizes_safety() -> None:
    """A zero false-safe target should select a safe threshold."""
    selected_threshold, search = select_feasibility_threshold(
        actual=[
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        probability=[
            0.20,
            0.55,
            0.80,
            0.60,
            0.70,
            0.90,
        ],
        thresholds=(
            0.50,
            0.60,
            0.85,
            0.95,
        ),
        max_false_safe_rate=0.0,
    )

    assert selected_threshold == pytest.approx(0.85)

    selected_row = search.loc[search["selected"]].iloc[0]

    assert selected_row["false_safe_rate"] == pytest.approx(0.0)

    assert bool(selected_row["meets_false_safe_target"])


def test_pipeline_creates_all_artifacts(
    tmp_path: Path,
) -> None:
    """The full classifier pipeline should create outputs."""
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "feasibility_classifier"

    _create_dataset(dataset_path)

    artifacts = train_feasibility_classifier(
        dataset_path=dataset_path,
        output_dir=output_dir,
        classifier_parameters={
            "n_estimators": 25,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
        threshold_grid=(
            0.30,
            0.50,
            0.70,
            0.85,
            0.95,
        ),
        max_false_safe_rate=0.10,
    )

    expected_paths = [
        artifacts.model_path,
        artifacts.threshold_search_path,
        artifacts.test_predictions_path,
        artifacts.test_metrics_path,
        artifacts.confusion_matrix_path,
        artifacts.probability_calibration_path,
        artifacts.summary_path,
    ]

    for path in expected_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    assert 0.0 <= (artifacts.selected_threshold) <= 1.0

    assert 0.0 <= (artifacts.false_safe_rate) <= 1.0

    assert 0.0 <= (artifacts.balanced_accuracy) <= 1.0

    assert artifacts.test_rows == 60

    predictions = pd.read_csv(artifacts.test_predictions_path)

    assert len(predictions) == 60

    assert {
        "is_feasible",
        "feasibility_probability",
        "infeasibility_probability",
        "predicted_is_feasible",
        "false_safe_prediction",
        "false_reject_prediction",
    }.issubset(predictions.columns)

    assert predictions["feasibility_probability"].between(0.0, 1.0).all()


def test_missing_split_is_rejected(
    tmp_path: Path,
) -> None:
    """Datasets missing a required split should fail."""
    dataset_path = tmp_path / "dataset.csv"

    _create_dataset(dataset_path)

    frame = pd.read_csv(dataset_path)

    frame = frame.loc[frame["split"] != "validation"]

    frame.to_csv(
        dataset_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required splits",
    ):
        train_feasibility_classifier(
            dataset_path=dataset_path,
            output_dir=tmp_path / "output",
        )


def test_invalid_false_safe_target_is_rejected() -> None:
    """The requested false-safe rate must be valid."""
    with pytest.raises(
        ValueError,
        match="max_false_safe_rate",
    ):
        select_feasibility_threshold(
            actual=[0, 1],
            probability=[0.20, 0.80],
            max_false_safe_rate=1.50,
        )

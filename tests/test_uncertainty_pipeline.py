"""Integration tests for uncertainty evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegenbench.uncertainty.pipeline import (
    evaluate_uncertainty,
)


def _create_dataset(path: Path) -> None:
    """Create a small deterministic benchmark dataset."""
    random_generator = np.random.default_rng(42)
    sample_count = 220

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

    architecture_code = (
        pd.Series(architectures)
        .map(
            {
                "conventional_turboprop": 0.0,
                "parallel_hybrid": 1.0,
                "series_hybrid": 2.0,
                "fuel_cell_electric": 3.0,
            }
        )
        .to_numpy()
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

    frame["estimated_takeoff_mass_kg"] = (
        9000.0
        + 90.0 * frame["passenger_capacity"]
        + 1.2 * frame["design_range_km"]
        + 300.0 * architecture_code
    )

    frame["mission_energy_kwh"] = (
        500.0
        + 8.0 * frame["design_range_km"]
        + 2.0 * frame["cruise_speed_kmh"]
        - 400.0 * frame["hybridization_ratio"]
        + 100.0 * architecture_code
    )

    frame["energy_per_passenger_km_kwh"] = (
        frame["mission_energy_kwh"] / frame["passenger_capacity"] / frame["design_range_km"]
    )

    frame["lifecycle_emissions_proxy_kgco2e"] = (
        0.15 * frame["mission_energy_kwh"] - 100.0 * architecture_code
    )

    frame["operating_cost_proxy_usd"] = (
        0.12 * frame["mission_energy_kwh"] + 0.02 * frame["estimated_takeoff_mass_kg"]
    )

    frame["noise_proxy_db"] = (
        78.0
        + 0.02 * frame["cruise_speed_kmh"]
        + 0.03 * frame["passenger_capacity"]
        - 2.0 * architecture_code
    )

    frame["split"] = ["train"] * 150 + ["validation"] * 35 + ["test"] * 35

    frame.to_csv(path, index=False)


def _create_random_forest_summary(
    path: Path,
) -> None:
    """Create a lightweight model-selection summary."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model_type": "random_forest",
        "best_parameters": {
            "n_estimators": 12,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
    }

    path.write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def test_uncertainty_pipeline_creates_outputs(
    tmp_path: Path,
) -> None:
    """The complete uncertainty pipeline should run."""
    dataset_path = tmp_path / "dataset.csv"
    summary_path = tmp_path / "tree_baselines" / "random_forest" / "summary.json"
    output_dir = tmp_path / "uncertainty"

    _create_dataset(dataset_path)
    _create_random_forest_summary(summary_path)

    artifacts = evaluate_uncertainty(
        dataset_path=dataset_path,
        random_forest_summary_path=summary_path,
        output_dir=output_dir,
        coverages=(0.80, 0.90),
        calibration_fraction=0.20,
    )

    assert artifacts.model_path.exists()
    assert artifacts.ensemble_intervals_path.exists()
    assert artifacts.conformal_quantiles_path.exists()
    assert artifacts.coverage_metrics_path.exists()
    assert artifacts.uncertainty_bins_path.exists()
    assert artifacts.summary_path.exists()

    assert len(artifacts.conformal_interval_paths) == 2

    for path in artifacts.conformal_interval_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    for path in artifacts.plot_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    coverage_metrics = pd.read_csv(artifacts.coverage_metrics_path)

    assert {
        "random_forest_tree_quantiles",
        "split_conformal",
    }.issubset(set(coverage_metrics["method"]))

    assert coverage_metrics["empirical_coverage"].between(0.0, 1.0).all()

    assert artifacts.calibration_rows == 30
    assert artifacts.test_rows == 35


def test_missing_random_forest_summary_is_rejected(
    tmp_path: Path,
) -> None:
    """A missing parameter summary should fail clearly."""
    dataset_path = tmp_path / "dataset.csv"
    _create_dataset(dataset_path)

    with pytest.raises(
        FileNotFoundError,
        match="Random-Forest summary",
    ):
        evaluate_uncertainty(
            dataset_path=dataset_path,
            random_forest_summary_path=(tmp_path / "missing.json"),
            output_dir=tmp_path / "uncertainty",
        )


def test_invalid_calibration_fraction_is_rejected(
    tmp_path: Path,
) -> None:
    """Invalid calibration fractions should fail."""
    dataset_path = tmp_path / "dataset.csv"
    summary_path = tmp_path / "summary.json"

    _create_dataset(dataset_path)
    _create_random_forest_summary(summary_path)

    with pytest.raises(
        ValueError,
        match="calibration_fraction",
    ):
        evaluate_uncertainty(
            dataset_path=dataset_path,
            random_forest_summary_path=summary_path,
            output_dir=tmp_path / "uncertainty",
            calibration_fraction=1.0,
        )

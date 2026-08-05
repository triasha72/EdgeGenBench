"""Integration tests for FP32 baseline training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgegenbench.models.fp32_linear import DEFAULT_TARGETS
from edgegenbench.training.fp32_baseline import train_fp32_baseline


def _create_dataset(path: Path) -> None:
    random_generator = np.random.default_rng(42)
    sample_count = 160

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

    frame["split"] = ["train"] * 100 + ["validation"] * 30 + ["test"] * 30

    frame.to_csv(path, index=False)


def test_training_pipeline_creates_all_artifacts(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "artifacts"

    _create_dataset(dataset_path)

    artifacts = train_fp32_baseline(
        dataset_path=dataset_path,
        output_dir=output_dir,
        alpha_grid=(1.0e-4, 1.0e-2, 1.0),
    )

    assert artifacts.model_path.exists()
    assert artifacts.validation_search_path.exists()
    assert artifacts.test_metrics_path.exists()
    assert artifacts.test_predictions_path.exists()
    assert artifacts.latency_path.exists()
    assert artifacts.summary_path.exists()

    metrics = pd.read_csv(artifacts.test_metrics_path)

    assert len(metrics) == len(DEFAULT_TARGETS)
    assert set(metrics["target"]) == set(DEFAULT_TARGETS)
    assert np.isfinite(metrics["rmse"]).all()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))

    assert summary["training_rows"] == 100
    assert summary["validation_rows"] == 30
    assert summary["test_rows"] == 30
    assert summary["best_alpha"] in {1.0e-4, 1.0e-2, 1.0}
    assert summary["model_size_bytes"] > 0

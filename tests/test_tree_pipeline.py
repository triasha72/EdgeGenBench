"""Integration tests for nonlinear baseline training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgegenbench.models.tree_surrogate import (
    HIST_GRADIENT_BOOSTING,
    RANDOM_FOREST,
)
from edgegenbench.training.tree_baselines import (
    train_tree_baselines,
)


def _create_test_dataset(path: Path) -> None:
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


def test_tree_pipeline_creates_artifacts(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "tree_artifacts"

    _create_test_dataset(dataset_path)

    model_grids = {
        RANDOM_FOREST: (
            {
                "n_estimators": 10,
                "max_depth": 6,
                "min_samples_leaf": 1,
            },
        ),
        HIST_GRADIENT_BOOSTING: (
            {
                "max_iter": 20,
                "max_leaf_nodes": 15,
                "learning_rate": 0.1,
            },
        ),
    }

    artifacts = train_tree_baselines(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_grids=model_grids,
    )

    assert len(artifacts) == 2

    for artifact in artifacts:
        assert artifact.model_path.exists()
        assert artifact.validation_search_path.exists()
        assert artifact.test_metrics_path.exists()
        assert artifact.test_predictions_path.exists()
        assert artifact.latency_path.exists()
        assert artifact.summary_path.exists()

        summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))

        assert summary["training_rows"] == 100
        assert summary["validation_rows"] == 30
        assert summary["test_rows"] == 30
        assert summary["model_size_bytes"] > 0
        assert np.isfinite(summary["mean_test_nrmse_std"])
        assert np.isfinite(summary["mean_test_r2"])

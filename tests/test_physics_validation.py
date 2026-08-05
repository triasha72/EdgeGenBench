"""Tests for physics validation of optimized designs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegenbench.evaluation.physics_validation import (
    validate_optimization_designs,
)
from edgegenbench.models.preprocessing import (
    FEATURE_COLUMNS,
)
from edgegenbench.physics.synthetic_aircraft import (
    simulate_designs,
)


def _write_config(
    path: Path,
) -> dict[str, float]:
    """Write benchmark constraints used by the physics model."""
    constraints = {
        "max_battery_mass_fraction": 0.32,
        "max_takeoff_mass_base_kg": 18000,
        "max_takeoff_mass_per_passenger_kg": 130,
        "max_hydrogen_tank_volume_base_m3": 5.0,
        "max_hydrogen_tank_volume_per_passenger_m3": 0.10,
    }

    path.write_text(
        """
dataset:
  seed: 42

constraints:
  max_battery_mass_fraction: 0.32
  max_takeoff_mass_base_kg: 18000
  max_takeoff_mass_per_passenger_kg: 130
  max_hydrogen_tank_volume_base_m3: 5.0
  max_hydrogen_tank_volume_per_passenger_m3: 0.10
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return constraints


def _create_designs(
    path: Path,
    constraints: dict[str, float],
) -> None:
    """Create designs whose surrogate values equal physics values."""
    designs = pd.DataFrame(
        {
            "passenger_capacity": [
                70,
                70,
                70,
                70,
            ],
            "design_range_km": [
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ],
            "cruise_speed_kmh": [
                460.0,
                500.0,
                540.0,
                580.0,
            ],
            ("battery_specific_energy_wh_per_kg"): [
                450.0,
                500.0,
                550.0,
                600.0,
            ],
            "hydrogen_storage_efficiency": [
                0.50,
                0.55,
                0.60,
                0.65,
            ],
            "hybridization_ratio": [
                0.0,
                0.25,
                0.40,
                0.20,
            ],
            "propulsion_architecture": [
                "conventional_turboprop",
                "parallel_hybrid",
                "series_hybrid",
                "fuel_cell_electric",
            ],
        }
    )

    outputs = simulate_designs(
        designs=designs.loc[
            :,
            list(FEATURE_COLUMNS),
        ],
        constraints=constraints,
        seed=42,
    )

    report = pd.concat(
        [
            designs,
            outputs,
        ],
        axis=1,
    )

    report.insert(
        0,
        "candidate_id",
        [
            "candidate_1",
            "candidate_2",
            "candidate_3",
            "candidate_4",
        ],
    )

    report.insert(
        0,
        "representative_role",
        [
            "low_cost",
            "low_emissions",
            "low_noise",
            "balanced",
        ],
    )

    report["predicted_is_feasible"] = report["is_feasible"]

    report.to_csv(
        path,
        index=False,
    )


def test_physics_validation_creates_outputs(
    tmp_path: Path,
) -> None:
    """Perfect surrogate values should have zero validation error."""
    config_path = tmp_path / "config.yaml"
    designs_path = tmp_path / "designs.csv"
    output_dir = tmp_path / "validation"

    constraints = _write_config(config_path)

    _create_designs(
        designs_path,
        constraints,
    )

    artifacts = validate_optimization_designs(
        designs_path=designs_path,
        benchmark_config_path=config_path,
        output_dir=output_dir,
    )

    assert artifacts.details_path.exists()
    assert artifacts.metrics_path.exists()
    assert artifacts.summary_path.exists()

    for plot_path in artifacts.plot_paths:
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    metrics = pd.read_csv(artifacts.metrics_path)

    assert np.allclose(
        metrics["mae"].to_numpy(dtype=np.float64),
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    assert np.allclose(
        metrics["rmse"].to_numpy(dtype=np.float64),
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    assert artifacts.design_count == 4

    assert artifacts.feasibility_agreement_rate == pytest.approx(1.0)


def test_missing_design_feature_is_rejected(
    tmp_path: Path,
) -> None:
    """Missing model inputs should produce a clear error."""
    config_path = tmp_path / "config.yaml"
    designs_path = tmp_path / "designs.csv"

    _write_config(config_path)

    pd.DataFrame(
        {
            "passenger_capacity": [70],
        }
    ).to_csv(
        designs_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing feature columns",
    ):
        validate_optimization_designs(
            designs_path=designs_path,
            benchmark_config_path=config_path,
            output_dir=tmp_path / "output",
        )

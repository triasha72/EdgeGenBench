"""Tests for optimization candidate generation."""

from pathlib import Path

import pandas as pd
import pytest

from edgegenbench.optimization.design_space import (
    generate_candidate_designs,
    load_optimization_config,
)


def _write_config(
    path: Path,
    *,
    cruise_minimum: float = 420.0,
    cruise_maximum: float = 650.0,
) -> None:
    """Write a lightweight optimization configuration."""
    path.write_text(
        f"""
schema_version: "1.0.0"

optimization:
  name: "test_optimization"
  n_candidates: 40
  seed: 42

mission:
  passenger_capacity: 70
  design_range_km: 1000

design_space:
  cruise_speed_kmh:
    min: {cruise_minimum}
    max: {cruise_maximum}

  battery_specific_energy_wh_per_kg:
    min: 300
    max: 750

  hydrogen_storage_efficiency:
    min: 0.45
    max: 0.70

  hybridization_ratio:
    min: 0.00
    max: 0.65

  propulsion_architecture:
    categories:
      - conventional_turboprop
      - parallel_hybrid
      - series_hybrid
      - fuel_cell_electric

objectives:
  - name: operating_cost_proxy_usd
    direction: minimize

  - name: lifecycle_emissions_proxy_kgco2e
    direction: minimize

output:
  directory: artifacts/test_optimization
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_candidate_generation_is_reproducible(
    tmp_path: Path,
) -> None:
    """The same seed should create identical candidates."""
    config_path = tmp_path / "optimization.yaml"
    _write_config(config_path)

    config = load_optimization_config(config_path)

    first_candidates = generate_candidate_designs(config)
    second_candidates = generate_candidate_designs(config)

    pd.testing.assert_frame_equal(
        first_candidates,
        second_candidates,
    )


def test_candidate_generation_respects_design_space(
    tmp_path: Path,
) -> None:
    """Generated candidates should respect mission and bounds."""
    config_path = tmp_path / "optimization.yaml"
    _write_config(config_path)

    config = load_optimization_config(config_path)

    candidates = generate_candidate_designs(config)

    assert len(candidates) == 40
    assert candidates["candidate_id"].is_unique

    assert (candidates["passenger_capacity"] == 70).all()

    assert (candidates["design_range_km"] == 1000.0).all()

    bounded_columns = {
        "cruise_speed_kmh": (420.0, 650.0),
        "battery_specific_energy_wh_per_kg": (
            300.0,
            750.0,
        ),
        "hydrogen_storage_efficiency": (
            0.45,
            0.70,
        ),
        "hybridization_ratio": (
            0.0,
            0.65,
        ),
    }

    for column, (
        minimum,
        maximum,
    ) in bounded_columns.items():
        assert (
            candidates[column]
            .between(
                minimum,
                maximum,
            )
            .all()
        )

    architecture_counts = candidates["propulsion_architecture"].value_counts()

    assert architecture_counts.max() - architecture_counts.min() <= 1

    conventional_candidates = candidates.loc[
        candidates["propulsion_architecture"] == "conventional_turboprop"
    ]

    assert (conventional_candidates["hybridization_ratio"] == 0.0).all()


def test_invalid_bounds_are_rejected(
    tmp_path: Path,
) -> None:
    """Minimum bounds must remain below maximum bounds."""
    config_path = tmp_path / "invalid.yaml"

    _write_config(
        config_path,
        cruise_minimum=650.0,
        cruise_maximum=420.0,
    )

    with pytest.raises(
        ValueError,
        match="minimum must be less than maximum",
    ):
        load_optimization_config(config_path)


def test_missing_config_is_rejected(
    tmp_path: Path,
) -> None:
    """Missing configuration files should fail clearly."""
    with pytest.raises(
        FileNotFoundError,
        match="Optimization config",
    ):
        load_optimization_config(tmp_path / "missing.yaml")

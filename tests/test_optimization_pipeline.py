"""Integration tests for multi-objective optimization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegenbench.evaluation.optimization import (
    select_representative_designs,
)
from edgegenbench.models.feasibility import (
    FeasibilityClassifier,
)
from edgegenbench.models.tree_surrogate import (
    RANDOM_FOREST,
    TreeSurrogate,
)
from edgegenbench.optimization.pipeline import (
    optimize_designs,
)

OBJECTIVES = (
    "lifecycle_emissions_proxy_kgco2e",
    "operating_cost_proxy_usd",
    "noise_proxy_db",
)


def _write_config(
    path: Path,
) -> None:
    """Write a lightweight optimization configuration."""
    path.write_text(
        """
schema_version: "1.0.0"

optimization:
  name: "test_multi_objective_optimization"
  n_candidates: 120
  seed: 42

mission:
  passenger_capacity: 70
  design_range_km: 900

design_space:
  cruise_speed_kmh:
    min: 420
    max: 650

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
  - name: lifecycle_emissions_proxy_kgco2e
    direction: minimize

  - name: operating_cost_proxy_usd
    direction: minimize

  - name: noise_proxy_db
    direction: minimize

output:
  directory: artifacts/test_optimization
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _make_training_frame(
    sample_count: int = 320,
) -> pd.DataFrame:
    """Create deterministic data for optimization models."""
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
            ("battery_specific_energy_wh_per_kg"): random_generator.uniform(
                300.0,
                750.0,
                size=sample_count,
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

    frame["lifecycle_emissions_proxy_kgco2e"] = (
        2800.0
        + 4.0 * frame["design_range_km"]
        - 420.0 * architecture_code
        - 300.0 * frame["hybridization_ratio"]
    )

    frame["operating_cost_proxy_usd"] = (
        900.0
        + 1.9 * frame["design_range_km"]
        + 4.5 * frame["passenger_capacity"]
        + 75.0 * architecture_code
    )

    frame["noise_proxy_db"] = (
        77.0
        + 0.018 * frame["cruise_speed_kmh"]
        + 0.025 * frame["passenger_capacity"]
        - 2.1 * architecture_code
    )

    frame["is_feasible"] = (
        (frame["design_range_km"] < 1250.0)
        & (frame["hybridization_ratio"] < 0.56)
        & (frame["battery_specific_energy_wh_per_kg"] > 335.0)
    )

    return frame


def _save_models(
    directory: Path,
) -> tuple[Path, Path]:
    """Fit and save lightweight optimization models."""
    frame = _make_training_frame()

    surrogate_model = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=OBJECTIVES,
        parameters={
            "n_estimators": 20,
            "max_depth": 9,
            "min_samples_leaf": 1,
        },
        random_state=42,
    )

    feasibility_model = FeasibilityClassifier.fit(
        frame=frame,
        parameters={
            "n_estimators": 20,
            "max_depth": 9,
            "min_samples_leaf": 1,
        },
        threshold=0.40,
        random_state=42,
    )

    surrogate_path = directory / "surrogate.joblib"
    feasibility_path = directory / "feasibility.joblib"

    surrogate_model.save(surrogate_path)
    feasibility_model.save(feasibility_path)

    return (
        surrogate_path,
        feasibility_path,
    )


def test_representative_design_selection() -> None:
    """Representative selection should include four roles."""
    pareto_front = pd.DataFrame(
        {
            "candidate_id": [
                "cost",
                "emissions",
                "noise",
                "balanced",
            ],
            ("lifecycle_emissions_proxy_kgco2e"): [
                5.0,
                1.0,
                4.0,
                2.5,
            ],
            "operating_cost_proxy_usd": [
                1.0,
                5.0,
                4.0,
                2.5,
            ],
            "noise_proxy_db": [
                5.0,
                4.0,
                1.0,
                2.5,
            ],
        }
    )

    representatives = select_representative_designs(
        pareto_front=pareto_front,
        objectives=OBJECTIVES,
        directions=(
            "minimize",
            "minimize",
            "minimize",
        ),
    )

    assert set(representatives["representative_role"]) == {
        "low_emissions",
        "low_cost",
        "low_noise",
        "balanced",
    }


def test_optimization_pipeline_creates_outputs(
    tmp_path: Path,
) -> None:
    """The complete optimization pipeline should run."""
    config_path = tmp_path / "optimization.yaml"
    output_dir = tmp_path / "output"

    _write_config(config_path)

    (
        surrogate_path,
        feasibility_path,
    ) = _save_models(tmp_path)

    artifacts = optimize_designs(
        config_path=config_path,
        surrogate_model_path=(surrogate_path),
        feasibility_model_path=(feasibility_path),
        output_dir=output_dir,
    )

    expected_paths = [
        artifacts.candidate_designs_path,
        artifacts.feasible_candidates_path,
        artifacts.pareto_front_path,
        artifacts.representative_designs_path,
        artifacts.summary_path,
    ]

    for path in expected_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    for path in artifacts.plot_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    assert artifacts.candidate_count == 120
    assert artifacts.feasible_count > 0
    assert artifacts.pareto_count > 0
    assert artifacts.representative_count == 4

    pareto_front = pd.read_csv(artifacts.pareto_front_path)

    assert (pareto_front["pareto_rank"] == 1).all()

    representatives = pd.read_csv(artifacts.representative_designs_path)

    assert len(representatives) == 4


def test_missing_surrogate_model_is_rejected(
    tmp_path: Path,
) -> None:
    """Missing trained models should fail clearly."""
    config_path = tmp_path / "optimization.yaml"

    _write_config(config_path)

    with pytest.raises(
        FileNotFoundError,
        match="Surrogate model",
    ):
        optimize_designs(
            config_path=config_path,
            surrogate_model_path=(tmp_path / "missing.joblib"),
            feasibility_model_path=(tmp_path / "missing_feasibility.joblib"),
            output_dir=tmp_path / "output",
        )

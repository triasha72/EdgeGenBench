"""Candidate-design generation for EdgeGenBench optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

CONTINUOUS_OPTIMIZATION_VARIABLES = (
    "cruise_speed_kmh",
    "battery_specific_energy_wh_per_kg",
    "hydrogen_storage_efficiency",
    "hybridization_ratio",
)

VALID_OBJECTIVE_DIRECTIONS = (
    "minimize",
    "maximize",
)


@dataclass(frozen=True)
class OptimizationConfig:
    """Validated configuration for one optimization experiment."""

    schema_version: str
    name: str
    candidate_count: int
    seed: int
    passenger_capacity: int
    design_range_km: float
    continuous_bounds: dict[str, tuple[float, float]]
    architectures: tuple[str, ...]
    objectives: tuple[str, ...]
    objective_directions: tuple[str, ...]
    output_directory: Path


def _require_mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    """Require a configuration value to be a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a YAML mapping.")

    return value


def _read_bounds(
    design_space: dict[str, Any],
    variable: str,
) -> tuple[float, float]:
    """Read and validate one continuous-variable range."""
    specification = _require_mapping(
        design_space.get(variable),
        f"design_space.{variable}",
    )

    if "min" not in specification or "max" not in specification:
        raise ValueError(f"design_space.{variable} must contain min and max.")

    minimum = float(specification["min"])
    maximum = float(specification["max"])

    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        raise ValueError(f"Bounds for {variable} must be finite.")

    if minimum >= maximum:
        raise ValueError(f"Invalid bounds for {variable}: minimum must be less than maximum.")

    return minimum, maximum


def _read_objectives(
    raw_objectives: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read objective names and optimization directions."""
    if not isinstance(raw_objectives, list):
        raise ValueError("objectives must contain a YAML list.")

    if len(raw_objectives) < 2:
        raise ValueError("At least two objectives are required.")

    objective_names: list[str] = []
    objective_directions: list[str] = []

    for objective_index, raw_objective in enumerate(
        raw_objectives,
        start=1,
    ):
        objective = _require_mapping(
            raw_objective,
            f"objectives[{objective_index}]",
        )

        name = str(objective.get("name", "")).strip()
        direction = str(objective.get("direction", "")).strip()

        if not name:
            raise ValueError("Every objective must contain a name.")

        if direction not in VALID_OBJECTIVE_DIRECTIONS:
            raise ValueError(f"Unsupported objective direction: {direction}")

        objective_names.append(name)
        objective_directions.append(direction)

    if len(set(objective_names)) != len(objective_names):
        raise ValueError("Objective names must be unique.")

    return (
        tuple(objective_names),
        tuple(objective_directions),
    )


def load_optimization_config(
    config_path: Path,
) -> OptimizationConfig:
    """Load and validate an optimization YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Optimization config does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    config = _require_mapping(
        raw_config,
        "Optimization configuration",
    )

    optimization = _require_mapping(
        config.get("optimization"),
        "optimization",
    )
    mission = _require_mapping(
        config.get("mission"),
        "mission",
    )
    design_space = _require_mapping(
        config.get("design_space"),
        "design_space",
    )

    name = str(optimization.get("name", "")).strip()

    if not name:
        raise ValueError("optimization.name cannot be empty.")

    candidate_count = int(optimization.get("n_candidates", 0))
    seed = int(optimization.get("seed", 42))

    if candidate_count < 8:
        raise ValueError("n_candidates must be at least 8.")

    passenger_capacity = int(mission.get("passenger_capacity", 0))
    design_range_km = float(mission.get("design_range_km", 0.0))

    if passenger_capacity <= 0:
        raise ValueError("passenger_capacity must be positive.")

    if not np.isfinite(design_range_km) or design_range_km <= 0.0:
        raise ValueError("design_range_km must be finite and positive.")

    continuous_bounds = {
        variable: _read_bounds(
            design_space=design_space,
            variable=variable,
        )
        for variable in CONTINUOUS_OPTIMIZATION_VARIABLES
    }

    architecture_specification = _require_mapping(
        design_space.get("propulsion_architecture"),
        "design_space.propulsion_architecture",
    )

    raw_architectures = architecture_specification.get("categories")

    if not isinstance(raw_architectures, list):
        raise ValueError("Propulsion architectures must be a list.")

    architectures = tuple(
        str(architecture).strip() for architecture in raw_architectures if str(architecture).strip()
    )

    if not architectures:
        raise ValueError("At least one propulsion architecture is required.")

    if len(set(architectures)) != len(architectures):
        raise ValueError("Propulsion architectures must be unique.")

    objectives, objective_directions = _read_objectives(config.get("objectives"))

    output = _require_mapping(
        config.get("output", {}),
        "output",
    )

    output_directory = Path(
        str(
            output.get(
                "directory",
                "artifacts/optimization",
            )
        )
    )

    return OptimizationConfig(
        schema_version=str(config.get("schema_version", "1.0.0")),
        name=name,
        candidate_count=candidate_count,
        seed=seed,
        passenger_capacity=passenger_capacity,
        design_range_km=design_range_km,
        continuous_bounds=continuous_bounds,
        architectures=architectures,
        objectives=objectives,
        objective_directions=objective_directions,
        output_directory=output_directory,
    )


def generate_candidate_designs(
    config: OptimizationConfig,
) -> pd.DataFrame:
    """Generate reproducible Latin-hypercube design candidates."""
    lower_bounds = [
        config.continuous_bounds[variable][0] for variable in CONTINUOUS_OPTIMIZATION_VARIABLES
    ]
    upper_bounds = [
        config.continuous_bounds[variable][1] for variable in CONTINUOUS_OPTIMIZATION_VARIABLES
    ]

    sampler = qmc.LatinHypercube(
        d=len(CONTINUOUS_OPTIMIZATION_VARIABLES),
        seed=config.seed,
    )

    unit_samples = sampler.random(n=config.candidate_count)

    scaled_samples = qmc.scale(
        unit_samples,
        lower_bounds,
        upper_bounds,
    )

    candidates = pd.DataFrame(
        scaled_samples,
        columns=CONTINUOUS_OPTIMIZATION_VARIABLES,
    )

    architecture_values = np.resize(
        np.asarray(
            config.architectures,
            dtype=object,
        ),
        config.candidate_count,
    )

    random_generator = np.random.default_rng(config.seed + 1)
    random_generator.shuffle(architecture_values)

    candidates["propulsion_architecture"] = architecture_values

    conventional_mask = candidates["propulsion_architecture"] == "conventional_turboprop"

    candidates.loc[
        conventional_mask,
        "hybridization_ratio",
    ] = 0.0

    candidates.insert(
        0,
        "design_range_km",
        config.design_range_km,
    )

    candidates.insert(
        0,
        "passenger_capacity",
        config.passenger_capacity,
    )

    candidate_ids = [
        f"candidate_{candidate_number:06d}"
        for candidate_number in range(
            1,
            config.candidate_count + 1,
        )
    ]

    candidates.insert(
        0,
        "candidate_id",
        candidate_ids,
    )

    ordered_columns = [
        "candidate_id",
        "passenger_capacity",
        "design_range_km",
        "cruise_speed_kmh",
        "battery_specific_energy_wh_per_kg",
        "hydrogen_storage_efficiency",
        "hybridization_ratio",
        "propulsion_architecture",
    ]

    candidates = candidates.loc[
        :,
        ordered_columns,
    ]

    if not np.isfinite(
        candidates.loc[
            :,
            CONTINUOUS_OPTIMIZATION_VARIABLES,
        ].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("Generated candidates contain nonfinite values.")

    return candidates

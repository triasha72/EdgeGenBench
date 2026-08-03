"""Reproducible synthetic dataset generation for EdgeGenBench."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

from edgegenbench.physics.synthetic_aircraft import simulate_designs

CONTINUOUS_VARIABLES = (
    "passenger_capacity",
    "design_range_km",
    "cruise_speed_kmh",
    "battery_specific_energy_wh_per_kg",
    "hydrogen_storage_efficiency",
    "hybridization_ratio",
)


@dataclass(frozen=True)
class DatasetArtifacts:
    """Paths and summary information produced by one generation run."""

    data_path: Path
    metadata_path: Path
    row_count: int
    feasible_fraction: float


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping at its top level.")

    for required_key in ("dataset", "design_space", "constraints"):
        if required_key not in config:
            raise ValueError(f"Configuration is missing required key: {required_key}")

    return config


def _bounds(config: Mapping[str, Any], variable: str) -> tuple[float, float]:
    specification = config["design_space"][variable]
    minimum = float(specification["min"])
    maximum = float(specification["max"])

    if minimum >= maximum:
        raise ValueError(f"Invalid bounds for {variable}: minimum must be less than maximum.")

    return minimum, maximum


def generate_design_samples(config: Mapping[str, Any]) -> pd.DataFrame:
    """Create balanced architecture samples with Latin-hypercube continuous inputs."""
    dataset_config = config["dataset"]
    sample_count = int(dataset_config["n_samples"])
    seed = int(dataset_config["seed"])

    if sample_count < 8:
        raise ValueError("n_samples must be at least 8.")

    lower_bounds, upper_bounds = zip(
        *(_bounds(config, variable) for variable in CONTINUOUS_VARIABLES),
        strict=True,
    )

    sampler = qmc.LatinHypercube(d=len(CONTINUOUS_VARIABLES), seed=seed)
    unit_samples = sampler.random(n=sample_count)
    scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)
    designs = pd.DataFrame(scaled_samples, columns=CONTINUOUS_VARIABLES)
    designs["passenger_capacity"] = designs["passenger_capacity"].round().astype(int)

    categories = config["design_space"]["propulsion_architecture"]["categories"]
    if not categories:
        raise ValueError("At least one propulsion architecture is required.")

    architecture_values = np.resize(np.asarray(categories, dtype=object), sample_count)
    random_generator = np.random.default_rng(seed)
    random_generator.shuffle(architecture_values)

    designs["propulsion_architecture"] = architecture_values
    designs.loc[
        designs["propulsion_architecture"] == "conventional_turboprop",
        "hybridization_ratio",
    ] = 0.0

    return designs


def _assign_splits(designs: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    dataset_config = config["dataset"]
    train_fraction = float(dataset_config["train_fraction"])
    validation_fraction = float(dataset_config["validation_fraction"])

    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("Train and validation fractions must each be between zero and one.")

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave samples for test.")

    split = pd.Series(index=designs.index, dtype="object")
    random_generator = np.random.default_rng(int(dataset_config["seed"]) + 1)

    for _, group in designs.groupby("propulsion_architecture", sort=False):
        indices = group.index.to_numpy(copy=True)
        random_generator.shuffle(indices)

        train_end = int(len(indices) * train_fraction)
        validation_end = train_end + int(len(indices) * validation_fraction)

        split.loc[indices[:train_end]] = "train"
        split.loc[indices[train_end:validation_end]] = "validation"
        split.loc[indices[validation_end:]] = "test"

    return split


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def generate_dataset(
    config_path: Path,
    output_dir: Path | None = None,
) -> DatasetArtifacts:
    """Generate a synthetic design dataset and reproducibility metadata."""
    config = _load_config(config_path)
    designs = generate_design_samples(config)

    outputs = simulate_designs(
        designs=designs,
        constraints=config["constraints"],
        seed=int(config["dataset"]["seed"]),
    )

    dataset = pd.concat([designs, outputs], axis=1)
    dataset["split"] = _assign_splits(designs, config)

    destination = output_dir or Path(config.get("output", {}).get("directory", "data/raw"))
    destination.mkdir(parents=True, exist_ok=True)

    data_path = destination / str(config["dataset"]["filename"])
    dataset.to_csv(data_path, index=False, float_format="%.10f")

    split_counts = {key: int(value) for key, value in dataset["split"].value_counts().items()}
    metadata = {
        "schema_version": config.get("schema_version", "1.0.0"),
        "dataset_name": config["dataset"]["name"],
        "row_count": int(len(dataset)),
        "seed": int(config["dataset"]["seed"]),
        "split_counts": split_counts,
        "feasible_fraction": float(dataset["is_feasible"].mean()),
        "columns": list(dataset.columns),
        "dataset_sha256": _file_sha256(data_path),
        "config": config,
    }

    metadata_path = data_path.with_name(f"{data_path.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return DatasetArtifacts(
        data_path=data_path,
        metadata_path=metadata_path,
        row_count=len(dataset),
        feasible_fraction=float(dataset["is_feasible"].mean()),
    )

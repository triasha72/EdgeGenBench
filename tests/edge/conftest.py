"""Shared test fixtures for edge deployment."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from edgegenbench.deployment.onnx_export import (
    export_edge_models,
)
from edgegenbench.models.feasibility import (
    FeasibilityClassifier,
)
from edgegenbench.models.tree_surrogate import (
    RANDOM_FOREST,
    TreeSurrogate,
)

TARGETS = (
    "lifecycle_emissions_proxy_kgco2e",
    "operating_cost_proxy_usd",
    "noise_proxy_db",
)


def _make_frame(
    sample_count: int = 160,
) -> pd.DataFrame:
    """Create deterministic edge-test data."""
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

    frame[TARGETS[0]] = (
        2500.0
        + 4.0 * frame["design_range_km"]
        - 350.0 * architecture_code
        - 220.0 * frame["hybridization_ratio"]
    )

    frame[TARGETS[1]] = (
        900.0
        + 1.8 * frame["design_range_km"]
        + 4.0 * frame["passenger_capacity"]
        + 70.0 * architecture_code
    )

    frame[TARGETS[2]] = 76.0 + 0.02 * frame["cruise_speed_kmh"] - 1.8 * architecture_code

    frame["is_feasible"] = (
        (frame["design_range_km"] < 1125.0)
        & (frame["hybridization_ratio"] < 0.50)
        & (frame["battery_specific_energy_wh_per_kg"] > 370.0)
    )

    frame["split"] = ["train"] * 112 + ["validation"] * 24 + ["test"] * 24

    return frame


@pytest.fixture(scope="session")
def edge_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Train and export lightweight test models."""
    root = tmp_path_factory.mktemp("edge_bundle")

    frame = _make_frame()

    surrogate = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=TARGETS,
        parameters={
            "n_estimators": 16,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
        random_state=42,
    )

    classifier = FeasibilityClassifier.fit(
        frame=frame,
        parameters={
            "n_estimators": 16,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
        threshold=0.60,
        random_state=42,
    )

    surrogate_path = root / "surrogate.joblib"
    classifier_path = root / "classifier.joblib"
    dataset_path = root / "dataset.csv"

    surrogate.save(surrogate_path)
    classifier.save(classifier_path)
    frame.to_csv(
        dataset_path,
        index=False,
    )

    export_artifacts = export_edge_models(
        surrogate_model_path=(surrogate_path),
        feasibility_model_path=(classifier_path),
        output_dir=root / "edge_export",
    )

    return {
        "root": root,
        "frame": frame,
        "surrogate": surrogate,
        "classifier": classifier,
        "surrogate_path": surrogate_path,
        "classifier_path": classifier_path,
        "dataset_path": dataset_path,
        "export": export_artifacts,
    }

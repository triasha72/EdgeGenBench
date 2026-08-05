"""Tests for surrogate-assisted optimization search."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgegenbench.models.feasibility import (
    FeasibilityClassifier,
)
from edgegenbench.models.tree_surrogate import (
    RANDOM_FOREST,
    TreeSurrogate,
)
from edgegenbench.optimization.search import (
    filter_feasible_candidates,
    run_candidate_search,
    run_candidate_search_from_paths,
    score_candidate_designs,
)

OBJECTIVES = (
    "lifecycle_emissions_proxy_kgco2e",
    "operating_cost_proxy_usd",
    "noise_proxy_db",
)


def _make_training_frame(
    sample_count: int = 180,
) -> pd.DataFrame:
    """Create deterministic optimization training data."""
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
            "propulsion_architecture": architectures,
        }
    )

    frame["lifecycle_emissions_proxy_kgco2e"] = (
        2500.0
        + 4.0 * frame["design_range_km"]
        - 400.0 * architecture_code
        - 250.0 * frame["hybridization_ratio"]
    )

    frame["operating_cost_proxy_usd"] = (
        1000.0
        + 1.8 * frame["design_range_km"]
        + 5.0 * frame["passenger_capacity"]
        + 80.0 * architecture_code
    )

    frame["noise_proxy_db"] = (
        75.0
        + 0.02 * frame["cruise_speed_kmh"]
        + 0.03 * frame["passenger_capacity"]
        - 2.0 * architecture_code
    )

    frame["is_feasible"] = (
        (frame["design_range_km"] < 1100.0)
        & (frame["hybridization_ratio"] < 0.48)
        & (frame["battery_specific_energy_wh_per_kg"] > 380.0)
    )

    return frame


def _make_models() -> tuple[
    TreeSurrogate,
    FeasibilityClassifier,
]:
    """Fit lightweight models for optimization tests."""
    frame = _make_training_frame()

    surrogate_model = TreeSurrogate.fit(
        frame=frame,
        model_type=RANDOM_FOREST,
        targets=OBJECTIVES,
        parameters={
            "n_estimators": 20,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
        random_state=42,
    )

    feasibility_model = FeasibilityClassifier.fit(
        frame=frame,
        parameters={
            "n_estimators": 20,
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
        threshold=0.65,
        random_state=42,
    )

    return surrogate_model, feasibility_model


def _make_candidates(
    count: int = 24,
) -> pd.DataFrame:
    """Create candidate designs from the training domain."""
    frame = _make_training_frame(sample_count=max(count, 24))

    candidates = (
        frame.iloc[:count]
        .loc[
            :,
            [
                "passenger_capacity",
                "design_range_km",
                "cruise_speed_kmh",
                "battery_specific_energy_wh_per_kg",
                "hydrogen_storage_efficiency",
                "hybridization_ratio",
                "propulsion_architecture",
            ],
        ]
        .reset_index(drop=True)
    )

    candidates.insert(
        0,
        "candidate_id",
        [f"candidate_{index:04d}" for index in range(count)],
    )

    return candidates


def test_candidate_scoring_adds_predictions() -> None:
    """Candidate scoring should append objectives and safety data."""
    surrogate_model, feasibility_model = _make_models()
    candidates = _make_candidates()

    scored = score_candidate_designs(
        candidates=candidates,
        surrogate_model=surrogate_model,
        feasibility_model=feasibility_model,
        objectives=OBJECTIVES,
    )

    assert len(scored) == len(candidates)

    expected_columns = {
        *OBJECTIVES,
        "feasibility_probability",
        "infeasibility_probability",
        "feasibility_threshold",
        "predicted_is_feasible",
    }

    assert expected_columns.issubset(scored.columns)

    assert scored["feasibility_probability"].between(0.0, 1.0).all()

    assert scored["infeasibility_probability"].between(0.0, 1.0).all()

    assert np.isfinite(
        scored.loc[
            :,
            list(OBJECTIVES),
        ].to_numpy()
    ).all()

    assert np.allclose(
        scored["feasibility_threshold"].to_numpy(dtype=np.float64),
        0.65,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_search_result_counts_feasible_candidates() -> None:
    """Search should report candidate and feasible counts."""
    surrogate_model, feasibility_model = _make_models()
    candidates = _make_candidates()

    result = run_candidate_search(
        candidates=candidates,
        surrogate_model=surrogate_model,
        feasibility_model=feasibility_model,
        objectives=OBJECTIVES,
    )

    assert result.candidate_count == len(candidates)

    assert result.feasible_count == len(result.feasible_candidates)

    assert result.feasibility_threshold == pytest.approx(0.65)

    assert result.feasible_fraction == pytest.approx(result.feasible_count / result.candidate_count)

    assert result.feasible_candidates["predicted_is_feasible"].all()


def test_feasible_filter_matches_prediction_flag() -> None:
    """Filtering should retain exactly accepted candidates."""
    surrogate_model, feasibility_model = _make_models()
    candidates = _make_candidates()

    scored = score_candidate_designs(
        candidates=candidates,
        surrogate_model=surrogate_model,
        feasibility_model=feasibility_model,
        objectives=OBJECTIVES,
    )

    feasible = filter_feasible_candidates(scored)

    expected_ids = set(
        scored.loc[
            scored["predicted_is_feasible"],
            "candidate_id",
        ]
    )

    assert set(feasible["candidate_id"]) == expected_ids


def test_saved_models_can_run_search(
    tmp_path: Path,
) -> None:
    """The path-based search should load serialized models."""
    surrogate_model, feasibility_model = _make_models()
    candidates = _make_candidates()

    surrogate_path = tmp_path / "surrogate.joblib"
    feasibility_path = tmp_path / "feasibility.joblib"

    surrogate_model.save(surrogate_path)
    feasibility_model.save(feasibility_path)

    result = run_candidate_search_from_paths(
        candidates=candidates,
        surrogate_model_path=surrogate_path,
        feasibility_model_path=feasibility_path,
        objectives=OBJECTIVES,
    )

    assert result.candidate_count == len(candidates)
    assert len(result.scored_candidates) == len(candidates)


def test_missing_surrogate_objective_is_rejected() -> None:
    """Requested objectives must exist in the surrogate."""
    surrogate_model, feasibility_model = _make_models()
    candidates = _make_candidates()

    with pytest.raises(
        ValueError,
        match="does not predict objectives",
    ):
        score_candidate_designs(
            candidates=candidates,
            surrogate_model=surrogate_model,
            feasibility_model=feasibility_model,
            objectives=(
                *OBJECTIVES,
                "missing_objective",
            ),
        )


def test_duplicate_candidate_ids_are_rejected() -> None:
    """Each candidate must have a unique identifier."""
    surrogate_model, feasibility_model = _make_models()
    candidates = _make_candidates()

    candidates.loc[
        1,
        "candidate_id",
    ] = candidates.loc[
        0,
        "candidate_id",
    ]

    with pytest.raises(
        ValueError,
        match="identifiers must be unique",
    ):
        score_candidate_designs(
            candidates=candidates,
            surrogate_model=surrogate_model,
            feasibility_model=feasibility_model,
            objectives=OBJECTIVES,
        )

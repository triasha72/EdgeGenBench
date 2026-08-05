"""Surrogate-assisted candidate search for EdgeGenBench."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from edgegenbench.models.feasibility import (
    FeasibilityClassifier,
)
from edgegenbench.models.preprocessing import (
    validate_feature_columns,
)
from edgegenbench.models.tree_surrogate import (
    TreeSurrogate,
)


@dataclass(frozen=True)
class CandidateSearchResult:
    """Results from scoring and filtering candidate designs."""

    scored_candidates: pd.DataFrame
    feasible_candidates: pd.DataFrame
    feasibility_threshold: float
    candidate_count: int
    feasible_count: int

    @property
    def feasible_fraction(self) -> float:
        """Return the fraction accepted by the classifier."""
        if self.candidate_count == 0:
            return 0.0

        return self.feasible_count / self.candidate_count


def _validate_candidate_frame(
    candidates: pd.DataFrame,
) -> None:
    """Validate generated optimization candidates."""
    if candidates.empty:
        raise ValueError("At least one candidate design is required.")

    validate_feature_columns(candidates)

    if "candidate_id" not in candidates.columns:
        raise ValueError("Candidate designs are missing candidate_id.")

    if candidates["candidate_id"].isna().any():
        raise ValueError("Candidate identifiers cannot be missing.")

    if not candidates["candidate_id"].is_unique:
        raise ValueError("Candidate identifiers must be unique.")


def _validate_objectives(
    surrogate_model: TreeSurrogate,
    objectives: Sequence[str],
) -> tuple[str, ...]:
    """Confirm that all objectives are predicted by the model."""
    objective_names = tuple(objectives)

    if not objective_names:
        raise ValueError("At least one optimization objective is required.")

    if len(set(objective_names)) != len(objective_names):
        raise ValueError("Optimization objectives must be unique.")

    missing_objectives = sorted(set(objective_names).difference(surrogate_model.targets))

    if missing_objectives:
        raise ValueError(f"Surrogate model does not predict objectives: {missing_objectives}")

    return objective_names


def load_search_models(
    surrogate_model_path: Path,
    feasibility_model_path: Path,
) -> tuple[
    TreeSurrogate,
    FeasibilityClassifier,
]:
    """Load the surrogate and feasibility models."""
    if not surrogate_model_path.exists():
        raise FileNotFoundError(f"Surrogate model does not exist: {surrogate_model_path}")

    if not feasibility_model_path.exists():
        raise FileNotFoundError(f"Feasibility model does not exist: {feasibility_model_path}")

    surrogate_model = TreeSurrogate.load(surrogate_model_path)

    feasibility_model = FeasibilityClassifier.load(feasibility_model_path)

    return surrogate_model, feasibility_model


def score_candidate_designs(
    candidates: pd.DataFrame,
    surrogate_model: TreeSurrogate,
    feasibility_model: FeasibilityClassifier,
    objectives: Sequence[str],
) -> pd.DataFrame:
    """Predict performance and feasibility for all candidates."""
    _validate_candidate_frame(candidates)

    objective_names = _validate_objectives(
        surrogate_model=surrogate_model,
        objectives=objectives,
    )

    overlapping_targets = sorted(set(surrogate_model.targets).intersection(candidates.columns))

    if overlapping_targets:
        raise ValueError(f"Candidate columns overlap surrogate targets: {overlapping_targets}")

    surrogate_predictions = surrogate_model.predict(candidates)

    feasibility_probability = feasibility_model.predict_feasibility_probability(candidates)

    if len(surrogate_predictions) != len(candidates):
        raise RuntimeError("Surrogate prediction row count does not match candidate count.")

    if len(feasibility_probability) != len(candidates):
        raise RuntimeError("Feasibility prediction row count does not match candidate count.")

    scored_candidates = candidates.reset_index(drop=True).copy()

    reset_predictions = surrogate_predictions.reset_index(drop=True)

    for target in surrogate_model.targets:
        scored_candidates[target] = reset_predictions[target].to_numpy(dtype=np.float64)

    probability_values = feasibility_probability.reset_index(drop=True).to_numpy(dtype=np.float64)

    threshold = float(feasibility_model.threshold)

    scored_candidates["feasibility_probability"] = probability_values

    scored_candidates["infeasibility_probability"] = 1.0 - probability_values

    scored_candidates["feasibility_threshold"] = threshold

    scored_candidates["predicted_is_feasible"] = probability_values >= threshold

    objective_values = scored_candidates.loc[
        :,
        list(objective_names),
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(objective_values).all():
        raise RuntimeError("Surrogate objective predictions contain nonfinite values.")

    if not np.isfinite(probability_values).all():
        raise RuntimeError("Feasibility probabilities contain nonfinite values.")

    if np.any((probability_values < 0.0) | (probability_values > 1.0)):
        raise RuntimeError("Feasibility probabilities are outside zero to one.")

    return scored_candidates


def filter_feasible_candidates(
    scored_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Return candidates accepted by the stored threshold."""
    required_columns = {
        "candidate_id",
        "feasibility_probability",
        "feasibility_threshold",
        "predicted_is_feasible",
    }

    missing_columns = sorted(required_columns.difference(scored_candidates.columns))

    if missing_columns:
        raise ValueError(f"Scored candidates are missing columns: {missing_columns}")

    feasible_candidates = scored_candidates.loc[
        scored_candidates["predicted_is_feasible"].astype(bool)
    ].copy()

    feasible_candidates = feasible_candidates.sort_values(
        by=[
            "feasibility_probability",
            "candidate_id",
        ],
        ascending=[
            False,
            True,
        ],
        ignore_index=True,
    )

    return feasible_candidates


def run_candidate_search(
    candidates: pd.DataFrame,
    surrogate_model: TreeSurrogate,
    feasibility_model: FeasibilityClassifier,
    objectives: Sequence[str],
) -> CandidateSearchResult:
    """Score candidates and apply the safety threshold."""
    scored_candidates = score_candidate_designs(
        candidates=candidates,
        surrogate_model=surrogate_model,
        feasibility_model=feasibility_model,
        objectives=objectives,
    )

    feasible_candidates = filter_feasible_candidates(scored_candidates)

    return CandidateSearchResult(
        scored_candidates=scored_candidates,
        feasible_candidates=feasible_candidates,
        feasibility_threshold=float(feasibility_model.threshold),
        candidate_count=len(scored_candidates),
        feasible_count=len(feasible_candidates),
    )


def run_candidate_search_from_paths(
    candidates: pd.DataFrame,
    surrogate_model_path: Path,
    feasibility_model_path: Path,
    objectives: Sequence[str],
    feasibility_threshold: float | None = None,
) -> CandidateSearchResult:
    """Load trained models and execute candidate search."""
    surrogate_model, feasibility_model = load_search_models(
        surrogate_model_path=surrogate_model_path,
        feasibility_model_path=(feasibility_model_path),
    )

    if feasibility_threshold is not None:
        feasibility_model = feasibility_model.with_threshold(feasibility_threshold)

    return run_candidate_search(
        candidates=candidates,
        surrogate_model=surrogate_model,
        feasibility_model=feasibility_model,
        objectives=objectives,
    )

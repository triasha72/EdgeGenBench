"""Pareto-front calculations for multi-objective optimization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

VALID_DIRECTIONS = {
    "minimize",
    "maximize",
}


def _prepare_objective_values(
    frame: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[str] | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
]:
    """Validate and transform objective values for minimization."""
    objective_names = tuple(objectives)

    if not objective_names:
        raise ValueError("At least one objective must be supplied.")

    if len(set(objective_names)) != len(objective_names):
        raise ValueError("Objective names must be unique.")

    missing_columns = sorted(set(objective_names).difference(frame.columns))

    if missing_columns:
        raise ValueError(f"Objective columns are missing: {missing_columns}")

    if directions is None:
        direction_names = tuple("minimize" for _ in objective_names)
    else:
        direction_names = tuple(directions)

    if len(direction_names) != len(objective_names):
        raise ValueError("Each objective requires one direction.")

    invalid_directions = sorted(set(direction_names).difference(VALID_DIRECTIONS))

    if invalid_directions:
        raise ValueError(f"Unsupported objective directions: {invalid_directions}")

    objective_values = frame.loc[
        :,
        list(objective_names),
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(objective_values).all():
        raise ValueError("Objective values must be finite.")

    transformed_values = objective_values.copy()

    for column_index, direction in enumerate(direction_names):
        if direction == "maximize":
            transformed_values[
                :,
                column_index,
            ] *= -1.0

    return (
        objective_names,
        direction_names,
        transformed_values,
    )


def calculate_pareto_mask(
    frame: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[str] | None = None,
) -> pd.Series:
    """Identify non-dominated rows.

    Duplicate objective vectors are retained because neither
    duplicate strictly dominates the other.
    """
    (
        _,
        _,
        objective_values,
    ) = _prepare_objective_values(
        frame=frame,
        objectives=objectives,
        directions=directions,
    )

    row_count = len(frame)

    if row_count == 0:
        return pd.Series(
            dtype=bool,
            index=frame.index,
            name="is_pareto",
        )

    sort_keys = tuple(
        objective_values[:, column_index]
        for column_index in reversed(range(objective_values.shape[1]))
    )

    processing_order = np.lexsort(sort_keys)

    front_indices: list[int] = []

    for candidate_index in processing_order:
        candidate_values = objective_values[candidate_index]

        if front_indices:
            front_index_array = np.asarray(
                front_indices,
                dtype=np.int64,
            )
            front_values = objective_values[front_index_array]

            candidate_is_dominated = np.any(
                np.all(
                    front_values <= candidate_values,
                    axis=1,
                )
                & np.any(
                    front_values < candidate_values,
                    axis=1,
                )
            )

            if candidate_is_dominated:
                continue

            candidate_dominates_front = np.all(
                candidate_values <= front_values,
                axis=1,
            ) & np.any(
                candidate_values < front_values,
                axis=1,
            )

            if np.any(candidate_dominates_front):
                front_indices = [
                    existing_index
                    for existing_index, is_dominated in zip(
                        front_indices,
                        candidate_dominates_front,
                        strict=True,
                    )
                    if not is_dominated
                ]

        front_indices.append(int(candidate_index))

    pareto_mask = np.zeros(
        row_count,
        dtype=bool,
    )
    pareto_mask[
        np.asarray(
            front_indices,
            dtype=np.int64,
        )
    ] = True

    return pd.Series(
        pareto_mask,
        index=frame.index,
        name="is_pareto",
        dtype=bool,
    )


def extract_pareto_front(
    frame: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a sorted copy of all non-dominated rows."""
    objective_names, direction_names, _ = _prepare_objective_values(
        frame=frame,
        objectives=objectives,
        directions=directions,
    )

    pareto_mask = calculate_pareto_mask(
        frame=frame,
        objectives=objective_names,
        directions=direction_names,
    )

    pareto_front = frame.loc[pareto_mask].copy()

    ascending = [direction == "minimize" for direction in direction_names]

    pareto_front = pareto_front.sort_values(
        by=list(objective_names),
        ascending=ascending,
        ignore_index=True,
    )

    pareto_front.insert(
        0,
        "pareto_rank",
        1,
    )

    return pareto_front

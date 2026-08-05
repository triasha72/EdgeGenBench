"""Evaluation and visualization for multi-objective optimization."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBJECTIVE_ROLE_NAMES = {
    "lifecycle_emissions_proxy_kgco2e": "low_emissions",
    "operating_cost_proxy_usd": "low_cost",
    "noise_proxy_db": "low_noise",
}


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    frame_name: str,
) -> None:
    """Validate required DataFrame columns."""
    missing_columns = sorted(set(columns).difference(frame.columns))

    if missing_columns:
        raise ValueError(f"{frame_name} is missing columns: {missing_columns}")


def _transform_objectives_for_minimization(
    frame: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[str],
) -> np.ndarray:
    """Transform all objectives into minimization form."""
    objective_names = tuple(objectives)
    direction_names = tuple(directions)

    if len(objective_names) != len(direction_names):
        raise ValueError("Each objective requires one direction.")

    _require_columns(
        frame=frame,
        columns=objective_names,
        frame_name="Optimization frame",
    )

    objective_values = frame.loc[
        :,
        list(objective_names),
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(objective_values).all():
        raise ValueError("Optimization objectives must be finite.")

    transformed_values = objective_values.copy()

    for column_index, direction in enumerate(direction_names):
        if direction == "maximize":
            transformed_values[
                :,
                column_index,
            ] *= -1.0
        elif direction != "minimize":
            raise ValueError(f"Unsupported objective direction: {direction}")

    return transformed_values


def select_representative_designs(
    pareto_front: pd.DataFrame,
    objectives: Sequence[str],
    directions: Sequence[str],
) -> pd.DataFrame:
    """Select objective-specific and balanced Pareto designs."""
    if pareto_front.empty:
        raise ValueError("The Pareto front cannot be empty.")

    if "representative_role" in pareto_front.columns or "selection_score" in pareto_front.columns:
        raise ValueError("Pareto data already contain representative columns.")

    objective_names = tuple(objectives)
    direction_names = tuple(directions)

    transformed_values = _transform_objectives_for_minimization(
        frame=pareto_front,
        objectives=objective_names,
        directions=direction_names,
    )

    representative_frames: list[pd.DataFrame] = []

    for objective_index, objective in enumerate(objective_names):
        objective_values = transformed_values[
            :,
            objective_index,
        ]

        best_position = int(np.argmin(objective_values))

        representative = pareto_front.iloc[[best_position]].copy()

        role_name = OBJECTIVE_ROLE_NAMES.get(
            objective,
            f"best_{objective}",
        )

        representative.insert(
            0,
            "representative_role",
            role_name,
        )

        representative.insert(
            1,
            "selection_score",
            float(objective_values[best_position]),
        )

        representative_frames.append(representative)

    normalized_values = np.zeros_like(
        transformed_values,
        dtype=np.float64,
    )

    for column_index in range(transformed_values.shape[1]):
        column_values = transformed_values[
            :,
            column_index,
        ]

        minimum = float(np.min(column_values))
        maximum = float(np.max(column_values))
        span = maximum - minimum

        if span > np.finfo(np.float64).eps:
            normalized_values[
                :,
                column_index,
            ] = (column_values - minimum) / span

    balanced_scores = np.mean(
        normalized_values,
        axis=1,
    )

    balanced_position = int(np.argmin(balanced_scores))

    balanced_design = pareto_front.iloc[[balanced_position]].copy()

    balanced_design.insert(
        0,
        "representative_role",
        "balanced",
    )

    balanced_design.insert(
        1,
        "selection_score",
        float(balanced_scores[balanced_position]),
    )

    representative_frames.append(balanced_design)

    representatives = pd.concat(
        representative_frames,
        ignore_index=True,
    )

    return representatives


def plot_objective_tradeoff(
    feasible_candidates: pd.DataFrame,
    pareto_front: pd.DataFrame,
    x_objective: str,
    y_objective: str,
    output_path: Path,
    title: str,
) -> Path:
    """Plot feasible candidates and their Pareto front."""
    required_columns = (
        x_objective,
        y_objective,
    )

    _require_columns(
        frame=feasible_candidates,
        columns=required_columns,
        frame_name="Feasible candidates",
    )

    _require_columns(
        frame=pareto_front,
        columns=required_columns,
        frame_name="Pareto front",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.scatter(
        feasible_candidates[x_objective],
        feasible_candidates[y_objective],
        s=10,
        alpha=0.20,
        label="Feasible candidates",
    )

    axis.scatter(
        pareto_front[x_objective],
        pareto_front[y_objective],
        s=32,
        label="Pareto front",
    )

    axis.set_title(title)
    axis.set_xlabel(x_objective)
    axis.set_ylabel(y_objective)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)

    return output_path


def plot_parallel_coordinates(
    reference_frame: pd.DataFrame,
    representative_designs: pd.DataFrame,
    columns: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot normalized representative-design trade-offs."""
    column_names = tuple(columns)

    if not column_names:
        raise ValueError("At least one parallel-coordinate column is required.")

    _require_columns(
        frame=reference_frame,
        columns=column_names,
        frame_name="Reference frame",
    )

    _require_columns(
        frame=representative_designs,
        columns=(
            "representative_role",
            *column_names,
        ),
        frame_name="Representative designs",
    )

    reference_values = reference_frame.loc[
        :,
        list(column_names),
    ].to_numpy(dtype=np.float64)

    representative_values = representative_designs.loc[
        :,
        list(column_names),
    ].to_numpy(dtype=np.float64)

    if not (np.isfinite(reference_values).all() and np.isfinite(representative_values).all()):
        raise ValueError("Parallel-coordinate values must be finite.")

    minimum_values = np.min(
        reference_values,
        axis=0,
    )

    maximum_values = np.max(
        reference_values,
        axis=0,
    )

    spans = maximum_values - minimum_values

    safe_spans = np.where(
        spans > np.finfo(np.float64).eps,
        spans,
        1.0,
    )

    normalized_values = (representative_values - minimum_values) / safe_spans

    constant_columns = spans <= np.finfo(np.float64).eps

    normalized_values[
        :,
        constant_columns,
    ] = 0.5

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(12, 6))

    x_positions = np.arange(len(column_names))

    for row_index, role in enumerate(representative_designs["representative_role"]):
        axis.plot(
            x_positions,
            normalized_values[row_index],
            marker="o",
            label=str(role),
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        column_names,
        rotation=25,
        ha="right",
    )

    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Normalized value within Pareto front")
    axis.set_title("Representative Pareto Designs")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)

    return output_path

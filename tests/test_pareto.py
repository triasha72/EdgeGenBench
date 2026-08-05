"""Tests for Pareto-front identification."""

import numpy as np
import pandas as pd
import pytest

from edgegenbench.optimization.pareto import (
    calculate_pareto_mask,
    extract_pareto_front,
)


def test_pareto_mask_identifies_non_dominated_rows() -> None:
    """Dominated rows should be excluded from the front."""
    frame = pd.DataFrame(
        {
            "design": [
                "low_cost",
                "balanced",
                "low_emissions",
                "low_noise",
                "dominated",
                "balanced_duplicate",
            ],
            "cost": [
                1.0,
                2.0,
                4.0,
                4.0,
                3.0,
                2.0,
            ],
            "emissions": [
                4.0,
                2.0,
                1.0,
                4.0,
                3.0,
                2.0,
            ],
            "noise": [
                4.0,
                2.0,
                4.0,
                1.0,
                3.0,
                2.0,
            ],
        }
    )

    mask = calculate_pareto_mask(
        frame=frame,
        objectives=(
            "cost",
            "emissions",
            "noise",
        ),
    )

    selected_designs = set(frame.loc[mask, "design"])

    assert selected_designs == {
        "low_cost",
        "balanced",
        "low_emissions",
        "low_noise",
        "balanced_duplicate",
    }


def test_extract_pareto_front_adds_rank() -> None:
    """Extracted Pareto designs should receive rank one."""
    frame = pd.DataFrame(
        {
            "design": ["a", "b", "c"],
            "cost": [1.0, 2.0, 3.0],
            "emissions": [3.0, 2.0, 3.0],
        }
    )

    pareto_front = extract_pareto_front(
        frame=frame,
        objectives=(
            "cost",
            "emissions",
        ),
    )

    assert set(pareto_front["design"]) == {"a", "b"}

    assert (pareto_front["pareto_rank"] == 1).all()


def test_mixed_objective_directions_are_supported() -> None:
    """Maximization and minimization may be combined."""
    frame = pd.DataFrame(
        {
            "design": ["a", "b", "c", "d"],
            "performance": [
                10.0,
                8.0,
                9.0,
                7.0,
            ],
            "cost": [
                5.0,
                4.0,
                6.0,
                7.0,
            ],
        }
    )

    mask = calculate_pareto_mask(
        frame=frame,
        objectives=(
            "performance",
            "cost",
        ),
        directions=(
            "maximize",
            "minimize",
        ),
    )

    assert set(frame.loc[mask, "design"]) == {"a", "b"}


def test_missing_objective_is_rejected() -> None:
    """Missing objective columns should fail clearly."""
    frame = pd.DataFrame(
        {
            "cost": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Objective columns are missing",
    ):
        calculate_pareto_mask(
            frame=frame,
            objectives=(
                "cost",
                "emissions",
            ),
        )


def test_nonfinite_objective_is_rejected() -> None:
    """Optimization objectives must contain finite values."""
    frame = pd.DataFrame(
        {
            "cost": [1.0, np.nan],
            "emissions": [2.0, 3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        calculate_pareto_mask(
            frame=frame,
            objectives=(
                "cost",
                "emissions",
            ),
        )

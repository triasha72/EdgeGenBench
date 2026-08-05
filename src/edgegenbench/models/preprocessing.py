"""Shared preprocessing for EdgeGenBench surrogate models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = (
    "passenger_capacity",
    "design_range_km",
    "cruise_speed_kmh",
    "battery_specific_energy_wh_per_kg",
    "hydrogen_storage_efficiency",
    "hybridization_ratio",
)

CATEGORICAL_FEATURES = ("propulsion_architecture",)

FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def validate_feature_columns(frame: pd.DataFrame) -> None:
    """Confirm that every required model input is available."""
    missing_columns = sorted(set(FEATURE_COLUMNS).difference(frame.columns))

    if missing_columns:
        raise ValueError(f"Input data are missing required feature columns: {missing_columns}")

    if frame.loc[:, list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Input features contain missing values.")


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing shared by nonlinear surrogate models."""
    categorical_encoder = OneHotEncoder(
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float32,
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                categorical_encoder,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

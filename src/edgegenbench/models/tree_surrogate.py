"""Nonlinear tree-based surrogate models for EdgeGenBench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from edgegenbench.models.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    validate_feature_columns,
)

RANDOM_FOREST = "random_forest"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"

SUPPORTED_MODEL_TYPES = (
    RANDOM_FOREST,
    HIST_GRADIENT_BOOSTING,
)


def _validate_targets(
    frame: pd.DataFrame,
    targets: Sequence[str],
) -> tuple[str, ...]:
    """Validate target columns and return a stable tuple."""
    target_names = tuple(targets)

    if not target_names:
        raise ValueError("At least one target must be supplied.")

    missing_targets = sorted(set(target_names).difference(frame.columns))

    if missing_targets:
        raise ValueError(f"Training data are missing target columns: {missing_targets}")

    if frame.loc[:, list(target_names)].isna().any().any():
        raise ValueError("Training targets contain missing values.")

    return target_names


def _build_estimator(
    model_type: str,
    parameters: Mapping[str, Any],
    random_state: int,
) -> RandomForestRegressor | MultiOutputRegressor:
    """Create the requested nonlinear estimator."""
    model_parameters = dict(parameters)

    if model_type == RANDOM_FOREST:
        return RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **model_parameters,
        )

    if model_type == HIST_GRADIENT_BOOSTING:
        base_estimator = HistGradientBoostingRegressor(
            random_state=random_state,
            **model_parameters,
        )

        return MultiOutputRegressor(
            estimator=base_estimator,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported model type: {model_type}. Supported values are: {SUPPORTED_MODEL_TYPES}"
    )


@dataclass
class TreeSurrogate:
    """Reusable wrapper around a fitted nonlinear surrogate pipeline."""

    model_type: str
    targets: tuple[str, ...]
    parameters: dict[str, Any]
    random_state: int
    pipeline: Pipeline

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        model_type: str,
        targets: Sequence[str],
        parameters: Mapping[str, Any] | None = None,
        random_state: int = 42,
    ) -> TreeSurrogate:
        """Fit one nonlinear multi-output surrogate."""
        validate_feature_columns(frame)

        target_names = _validate_targets(
            frame=frame,
            targets=targets,
        )
        model_parameters = dict(parameters or {})

        preprocessor = build_preprocessor()

        estimator = _build_estimator(
            model_type=model_type,
            parameters=model_parameters,
            random_state=random_state,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ]
        )

        features = frame.loc[:, list(FEATURE_COLUMNS)]

        target_values = frame.loc[:, list(target_names)].to_numpy(dtype=np.float64)

        pipeline.fit(features, target_values)

        return cls(
            model_type=model_type,
            targets=target_names,
            parameters=model_parameters,
            random_state=random_state,
            pipeline=pipeline,
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict all configured targets."""
        validate_feature_columns(frame)

        features = frame.loc[:, list(FEATURE_COLUMNS)]

        predictions = np.asarray(
            self.pipeline.predict(features),
            dtype=np.float64,
        )

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        if predictions.shape[1] != len(self.targets):
            raise RuntimeError("Prediction output does not match the configured targets.")

        return pd.DataFrame(
            predictions,
            columns=self.targets,
            index=frame.index,
        )

    def save(self, path: Path) -> None:
        """Serialize the fitted model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> TreeSurrogate:
        """Load a serialized nonlinear surrogate."""
        loaded_model = joblib.load(path)

        if not isinstance(loaded_model, cls):
            raise TypeError(f"Serialized object is not a {cls.__name__}: {path}")

        return loaded_model

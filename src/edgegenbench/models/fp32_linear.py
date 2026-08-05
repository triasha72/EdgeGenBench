"""Deterministic FP32 linear surrogate baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "passenger_capacity",
    "design_range_km",
    "cruise_speed_kmh",
    "battery_specific_energy_wh_per_kg",
    "hydrogen_storage_efficiency",
    "hybridization_ratio",
)

CATEGORICAL_FEATURE = "propulsion_architecture"

DEFAULT_TARGETS = (
    "estimated_takeoff_mass_kg",
    "mission_energy_kwh",
    "energy_per_passenger_km_kwh",
    "lifecycle_emissions_proxy_kgco2e",
    "operating_cost_proxy_usd",
    "noise_proxy_db",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise an error when required columns are unavailable."""
    missing = sorted(set(columns).difference(frame.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _encode_categories(
    values: pd.Series,
    categories: tuple[str, ...],
) -> np.ndarray:
    """One-hot encode propulsion architectures as FP32 values."""
    category_to_index = {category: index for index, category in enumerate(categories)}

    encoded_indices: list[int] = []

    for value in values.astype(str):
        if value not in category_to_index:
            raise ValueError(f"Unknown {CATEGORICAL_FEATURE} value: {value}")

        encoded_indices.append(category_to_index[value])

    identity = np.eye(len(categories), dtype=np.float32)

    return identity[np.asarray(encoded_indices, dtype=np.int64)]


@dataclass(frozen=True)
class FP32LinearSurrogate:
    """Multi-output ridge-regression surrogate using FP32 arrays."""

    numeric_features: tuple[str, ...]
    categories: tuple[str, ...]
    targets: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    coefficients: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        targets: Sequence[str] = DEFAULT_TARGETS,
        alpha: float = 1.0,
    ) -> FP32LinearSurrogate:
        """Fit a deterministic multi-output ridge-regression model."""
        if alpha <= 0.0:
            raise ValueError("alpha must be greater than zero.")

        target_names = tuple(targets)

        required_columns = (
            *NUMERIC_FEATURES,
            CATEGORICAL_FEATURE,
            *target_names,
        )
        _require_columns(frame, required_columns)

        categories = tuple(sorted(frame[CATEGORICAL_FEATURE].astype(str).unique()))

        if not categories:
            raise ValueError("At least one propulsion architecture is required.")

        numeric_values = frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)

        feature_mean = numeric_values.mean(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        feature_scale = numeric_values.std(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        feature_scale = np.where(
            feature_scale > np.float32(1.0e-8),
            feature_scale,
            np.float32(1.0),
        ).astype(np.float32)

        standardized_numeric = (numeric_values - feature_mean) / feature_scale

        one_hot_architecture = _encode_categories(
            frame[CATEGORICAL_FEATURE],
            categories,
        )

        feature_matrix = np.concatenate(
            [standardized_numeric, one_hot_architecture],
            axis=1,
        ).astype(np.float32)

        target_values = frame.loc[:, target_names].to_numpy(dtype=np.float32)

        target_mean = target_values.mean(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        target_scale = target_values.std(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        target_scale = np.where(
            target_scale > np.float32(1.0e-8),
            target_scale,
            np.float32(1.0),
        ).astype(np.float32)

        standardized_targets = (target_values - target_mean) / target_scale

        bias = np.ones(
            (len(feature_matrix), 1),
            dtype=np.float32,
        )

        design_matrix = np.concatenate(
            [feature_matrix, bias],
            axis=1,
        ).astype(np.float32)

        gram_matrix = design_matrix.T @ design_matrix
        right_hand_side = design_matrix.T @ standardized_targets

        penalty = np.eye(
            gram_matrix.shape[0],
            dtype=np.float32,
        )

        # Do not regularize the bias coefficient.
        penalty[-1, -1] = np.float32(0.0)

        coefficients = np.linalg.solve(
            gram_matrix + np.float32(alpha) * penalty,
            right_hand_side,
        ).astype(np.float32)

        return cls(
            numeric_features=NUMERIC_FEATURES,
            categories=categories,
            targets=target_names,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            coefficients=coefficients,
            alpha=float(alpha),
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict all configured target variables."""
        required_columns = (
            *self.numeric_features,
            CATEGORICAL_FEATURE,
        )
        _require_columns(frame, required_columns)

        numeric_values = frame.loc[:, self.numeric_features].to_numpy(dtype=np.float32)

        standardized_numeric = (numeric_values - self.feature_mean) / self.feature_scale

        one_hot_architecture = _encode_categories(
            frame[CATEGORICAL_FEATURE],
            self.categories,
        )

        feature_matrix = np.concatenate(
            [standardized_numeric, one_hot_architecture],
            axis=1,
        ).astype(np.float32)

        bias = np.ones(
            (len(feature_matrix), 1),
            dtype=np.float32,
        )

        design_matrix = np.concatenate(
            [feature_matrix, bias],
            axis=1,
        ).astype(np.float32)

        standardized_predictions = (design_matrix @ self.coefficients).astype(np.float32)

        predictions = (standardized_predictions * self.target_scale + self.target_mean).astype(
            np.float32
        )

        return pd.DataFrame(
            predictions,
            columns=self.targets,
            index=frame.index,
        )

    def save(self, path: Path) -> None:
        """Save the model in a compressed, non-pickle NumPy format."""
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            numeric_features=np.asarray(self.numeric_features),
            categories=np.asarray(self.categories),
            targets=np.asarray(self.targets),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            coefficients=self.coefficients,
            alpha=np.asarray(self.alpha, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> FP32LinearSurrogate:
        """Load a previously saved model."""
        with np.load(path, allow_pickle=False) as model_data:
            return cls(
                numeric_features=tuple(
                    str(value) for value in model_data["numeric_features"].tolist()
                ),
                categories=tuple(str(value) for value in model_data["categories"].tolist()),
                targets=tuple(str(value) for value in model_data["targets"].tolist()),
                feature_mean=model_data["feature_mean"].astype(np.float32),
                feature_scale=model_data["feature_scale"].astype(np.float32),
                target_mean=model_data["target_mean"].astype(np.float32),
                target_scale=model_data["target_scale"].astype(np.float32),
                coefficients=model_data["coefficients"].astype(np.float32),
                alpha=float(model_data["alpha"]),
            )

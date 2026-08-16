"""Leakage-safe preprocessing for the PyTorch neural surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    DEFAULT_TARGETS,
    NUMERIC_FEATURES,
)


@dataclass(frozen=True)
class NeuralPreprocessor:
    """Frozen preprocessing statistics fitted on training data only."""

    categories: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    targets: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        """Return encoded neural-network input width."""
        return len(NUMERIC_FEATURES) + len(self.categories)

    @property
    def output_dim(self) -> int:
        """Return neural-network output width."""
        return len(self.targets)

    @classmethod
    def fit(
        cls,
        training_frame: pd.DataFrame,
        targets: tuple[str, ...] = DEFAULT_TARGETS,
    ) -> NeuralPreprocessor:
        """Fit preprocessing statistics using training data only."""
        if training_frame.empty:
            raise ValueError("Training frame cannot be empty.")

        required_columns = {
            *NUMERIC_FEATURES,
            CATEGORICAL_FEATURE,
            *targets,
        }

        missing_columns = sorted(required_columns.difference(training_frame.columns))

        if missing_columns:
            raise ValueError(f"Training data are missing required columns: {missing_columns}")

        if training_frame.loc[:, list(required_columns)].isna().any().any():
            raise ValueError("Training data contain missing values.")

        categories = tuple(sorted(training_frame[CATEGORICAL_FEATURE].astype(str).unique()))

        if not categories:
            raise ValueError("At least one propulsion architecture is required.")

        numeric_values = training_frame.loc[
            :,
            NUMERIC_FEATURES,
        ].to_numpy(dtype=np.float32)

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

        target_values = training_frame.loc[
            :,
            list(targets),
        ].to_numpy(dtype=np.float32)

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

        return cls(
            categories=categories,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            targets=targets,
        )

    def transform_features(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Transform raw inputs into normalized FP32 model features."""
        required_columns = {
            *NUMERIC_FEATURES,
            CATEGORICAL_FEATURE,
        }

        missing_columns = sorted(required_columns.difference(frame.columns))

        if missing_columns:
            raise ValueError(f"Input data are missing required columns: {missing_columns}")

        numeric_values = frame.loc[
            :,
            NUMERIC_FEATURES,
        ].to_numpy(dtype=np.float32)

        standardized_numeric = (numeric_values - self.feature_mean) / self.feature_scale

        category_to_index = {category: index for index, category in enumerate(self.categories)}

        encoded_indices: list[int] = []

        for value in frame[CATEGORICAL_FEATURE].astype(str):
            if value not in category_to_index:
                raise ValueError(f"Unknown {CATEGORICAL_FEATURE} value: {value}")

            encoded_indices.append(category_to_index[value])

        one_hot = np.eye(
            len(self.categories),
            dtype=np.float32,
        )[
            np.asarray(
                encoded_indices,
                dtype=np.int64,
            )
        ]

        return np.concatenate(
            [
                standardized_numeric,
                one_hot,
            ],
            axis=1,
        ).astype(np.float32)

    def transform_targets(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Normalize regression targets."""
        targets = frame.loc[
            :,
            list(self.targets),
        ].to_numpy(dtype=np.float32)

        return ((targets - self.target_mean) / self.target_scale).astype(np.float32)

    def inverse_transform_targets(
        self,
        normalized_targets: np.ndarray,
    ) -> np.ndarray:
        """Restore normalized predictions to physical target units."""
        values = np.asarray(
            normalized_targets,
            dtype=np.float32,
        )

        if values.ndim != 2:
            raise ValueError("Expected normalized targets with two dimensions.")

        if values.shape[1] != self.output_dim:
            raise ValueError("Target dimension does not match preprocessor configuration.")

        return (values * self.target_scale + self.target_mean).astype(np.float32)

    def save(
        self,
        path: Path,
    ) -> None:
        """Persist frozen preprocessing statistics."""
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            categories=np.asarray(self.categories),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            targets=np.asarray(self.targets),
        )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> NeuralPreprocessor:
        """Load frozen preprocessing statistics."""
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact does not exist: {path}")

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            return cls(
                categories=tuple(str(value) for value in data["categories"].tolist()),
                feature_mean=data["feature_mean"].astype(np.float32),
                feature_scale=data["feature_scale"].astype(np.float32),
                target_mean=data["target_mean"].astype(np.float32),
                target_scale=data["target_scale"].astype(np.float32),
                targets=tuple(str(value) for value in data["targets"].tolist()),
            )

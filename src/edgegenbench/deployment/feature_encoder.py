"""Deterministic edge feature encoding for EdgeGenBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from edgegenbench.models.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    validate_feature_columns,
)


@dataclass(frozen=True)
class EdgeFeatureEncoder:
    """NumPy implementation of the fitted training preprocessor."""

    numeric_features: tuple[str, ...]
    categorical_feature: str
    categories: tuple[str, ...]
    transformed_feature_names: tuple[str, ...]

    @classmethod
    def from_fitted_preprocessor(
        cls,
        preprocessor: ColumnTransformer,
    ) -> EdgeFeatureEncoder:
        """Build an edge encoder from a fitted preprocessor."""
        if not hasattr(
            preprocessor,
            "transformers_",
        ):
            raise ValueError("The preprocessing transformer has not been fitted.")

        if len(CATEGORICAL_FEATURES) != 1:
            raise ValueError("The edge encoder currently supports exactly one categorical feature.")

        categorical_transformer = preprocessor.named_transformers_["categorical"]

        if not isinstance(
            categorical_transformer,
            OneHotEncoder,
        ):
            raise TypeError("The categorical transformer must be a OneHotEncoder.")

        if len(categorical_transformer.categories_) != 1:
            raise ValueError("Expected one fitted categorical feature.")

        categories = tuple(str(category) for category in (categorical_transformer.categories_[0]))

        transformed_feature_names = tuple(
            str(feature_name) for feature_name in (preprocessor.get_feature_names_out())
        )

        expected_feature_count = len(NUMERIC_FEATURES) + len(categories)

        if len(transformed_feature_names) != expected_feature_count:
            raise RuntimeError("Unexpected transformed feature count.")

        return cls(
            numeric_features=tuple(NUMERIC_FEATURES),
            categorical_feature=(CATEGORICAL_FEATURES[0]),
            categories=categories,
            transformed_feature_names=(transformed_feature_names),
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
    ) -> EdgeFeatureEncoder:
        """Restore an encoder from serialized metadata."""
        return cls(
            numeric_features=tuple(str(value) for value in metadata["numeric_features"]),
            categorical_feature=str(metadata["categorical_feature"]),
            categories=tuple(str(value) for value in metadata["categories"]),
            transformed_feature_names=tuple(
                str(value) for value in metadata["transformed_feature_names"]
            ),
        )

    @property
    def feature_count(self) -> int:
        """Return the encoded feature width."""
        return len(self.transformed_feature_names)

    def to_metadata(
        self,
    ) -> dict[str, Any]:
        """Serialize the encoder configuration."""
        return {
            "numeric_features": list(self.numeric_features),
            "categorical_feature": (self.categorical_feature),
            "categories": list(self.categories),
            "transformed_feature_names": list(self.transformed_feature_names),
            "feature_count": (self.feature_count),
        }

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Encode raw design inputs as one FP32 tensor."""
        validate_feature_columns(frame)

        numeric_values = frame.loc[
            :,
            list(self.numeric_features),
        ].to_numpy(
            dtype=np.float32,
        )

        category_values = frame[self.categorical_feature].astype(str).to_numpy()

        category_to_index = {
            category: category_index for category_index, category in enumerate(self.categories)
        }

        unknown_categories = sorted(set(category_values).difference(category_to_index))

        if unknown_categories:
            raise ValueError(f"Unknown propulsion architectures: {unknown_categories}")

        one_hot_values = np.zeros(
            (
                len(frame),
                len(self.categories),
            ),
            dtype=np.float32,
        )

        row_indices = np.arange(
            len(frame),
            dtype=np.int64,
        )

        category_indices = np.asarray(
            [category_to_index[value] for value in category_values],
            dtype=np.int64,
        )

        one_hot_values[
            row_indices,
            category_indices,
        ] = 1.0

        encoded_values = np.concatenate(
            [
                numeric_values,
                one_hot_values,
            ],
            axis=1,
        )

        if encoded_values.shape[1] != self.feature_count:
            raise RuntimeError("Encoded feature width does not match metadata.")

        if not np.isfinite(encoded_values).all():
            raise ValueError("Encoded features contain nonfinite values.")

        return np.ascontiguousarray(
            encoded_values,
            dtype=np.float32,
        )

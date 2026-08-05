"""Random-Forest classifier for aircraft-design feasibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from edgegenbench.models.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    validate_feature_columns,
)

FEASIBILITY_TARGET = "is_feasible"

DEFAULT_CLASSIFIER_PARAMETERS: dict[str, Any] = {
    "n_estimators": 250,
    "max_depth": 18,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
}


def _validate_threshold(threshold: float) -> None:
    """Validate a feasibility decision threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one.")


def _validate_training_frame(
    frame: pd.DataFrame,
) -> pd.Series:
    """Validate classifier inputs and return binary targets."""
    validate_feature_columns(frame)

    if FEASIBILITY_TARGET not in frame.columns:
        raise ValueError(f"Training data are missing target: {FEASIBILITY_TARGET}")

    if frame[FEASIBILITY_TARGET].isna().any():
        raise ValueError("Feasibility targets contain missing values.")

    target_values = frame[FEASIBILITY_TARGET].astype(np.int64)

    available_classes = set(target_values.unique().tolist())

    if not available_classes.issubset({0, 1}):
        raise ValueError("Feasibility targets must be binary.")

    if available_classes != {0, 1}:
        raise ValueError("Training data must contain both feasible and infeasible designs.")

    return target_values


@dataclass
class FeasibilityClassifier:
    """Reusable feasibility classifier and decision threshold."""

    threshold: float
    parameters: dict[str, Any]
    random_state: int
    pipeline: Pipeline

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        parameters: Mapping[str, Any] | None = None,
        threshold: float = 0.50,
        random_state: int = 42,
    ) -> FeasibilityClassifier:
        """Fit a Random-Forest feasibility classifier."""
        _validate_threshold(threshold)

        target_values = _validate_training_frame(frame)

        supplied_parameters = dict(parameters or {})

        reserved_parameters = {
            "random_state",
            "n_jobs",
        }.intersection(supplied_parameters)

        if reserved_parameters:
            raise ValueError(
                f"Classifier parameters cannot override: {sorted(reserved_parameters)}"
            )

        model_parameters = dict(DEFAULT_CLASSIFIER_PARAMETERS)
        model_parameters.update(supplied_parameters)

        preprocessor = build_preprocessor()

        classifier = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **model_parameters,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        features = frame.loc[
            :,
            list(FEATURE_COLUMNS),
        ]

        pipeline.fit(
            features,
            target_values.to_numpy(),
        )

        return cls(
            threshold=float(threshold),
            parameters=model_parameters,
            random_state=random_state,
            pipeline=pipeline,
        )

    def predict_feasibility_probability(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Predict the probability that each design is feasible."""
        validate_feature_columns(frame)

        features = frame.loc[
            :,
            list(FEATURE_COLUMNS),
        ]

        classifier = self.pipeline.named_steps["classifier"]

        if not isinstance(
            classifier,
            RandomForestClassifier,
        ):
            raise TypeError("The fitted estimator is not a RandomForestClassifier.")

        probabilities = np.asarray(
            self.pipeline.predict_proba(features),
            dtype=np.float64,
        )

        class_labels = np.asarray(
            classifier.classes_,
            dtype=np.int64,
        )

        feasible_class_indices = np.flatnonzero(class_labels == 1)

        if len(feasible_class_indices) != 1:
            raise RuntimeError("The classifier does not contain a unique feasible class.")

        feasible_probability = probabilities[
            :,
            int(feasible_class_indices[0]),
        ]

        return pd.Series(
            feasible_probability,
            index=frame.index,
            name="feasibility_probability",
            dtype=np.float64,
        )

    def predict(
        self,
        frame: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.Series:
        """Predict whether each design should be accepted."""
        decision_threshold = self.threshold if threshold is None else float(threshold)

        _validate_threshold(decision_threshold)

        probabilities = self.predict_feasibility_probability(frame)

        predictions = probabilities >= decision_threshold

        return pd.Series(
            predictions,
            index=frame.index,
            name="predicted_is_feasible",
            dtype=bool,
        )

    def with_threshold(
        self,
        threshold: float,
    ) -> FeasibilityClassifier:
        """Create a classifier using a new decision threshold."""
        _validate_threshold(threshold)

        return FeasibilityClassifier(
            threshold=float(threshold),
            parameters=dict(self.parameters),
            random_state=self.random_state,
            pipeline=self.pipeline,
        )

    def save(self, path: Path) -> None:
        """Serialize the fitted classifier."""
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        joblib.dump(self, path)

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> FeasibilityClassifier:
        """Load a serialized feasibility classifier."""
        loaded_model = joblib.load(path)

        if not isinstance(
            loaded_model,
            cls,
        ):
            raise TypeError(f"Serialized object is not a {cls.__name__}: {path}")

        return loaded_model

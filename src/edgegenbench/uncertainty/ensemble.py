"""Random-Forest ensemble uncertainty for EdgeGenBench."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from edgegenbench.models.preprocessing import (
    FEATURE_COLUMNS,
    validate_feature_columns,
)
from edgegenbench.models.tree_surrogate import (
    RANDOM_FOREST,
    TreeSurrogate,
)


def _validate_coverage(coverage: float) -> None:
    """Validate a requested interval coverage."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one.")


def _tree_prediction_tensor(
    model: TreeSurrogate,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Return predictions from every tree.

    The returned array has shape:

    n_trees x n_samples x n_targets
    """
    if model.model_type != RANDOM_FOREST:
        raise ValueError("Tree ensemble uncertainty requires a Random Forest model.")

    validate_feature_columns(frame)

    preprocessor = model.pipeline.named_steps["preprocessor"]
    estimator = model.pipeline.named_steps["estimator"]

    if not isinstance(estimator, RandomForestRegressor):
        raise TypeError("The fitted estimator is not a RandomForestRegressor.")

    features = frame.loc[:, list(FEATURE_COLUMNS)]
    transformed_features = preprocessor.transform(features)

    tree_predictions: list[np.ndarray] = []

    for tree in estimator.estimators_:
        predictions = np.asarray(
            tree.predict(transformed_features),
            dtype=np.float64,
        )

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        tree_predictions.append(predictions)

    if not tree_predictions:
        raise RuntimeError("The Random Forest does not contain fitted trees.")

    prediction_tensor = np.stack(
        tree_predictions,
        axis=0,
    )

    if prediction_tensor.shape[2] != len(model.targets):
        raise RuntimeError("Tree predictions do not match the configured targets.")

    return prediction_tensor


def predict_tree_ensemble_intervals(
    model: TreeSurrogate,
    frame: pd.DataFrame,
    coverage: float = 0.90,
) -> pd.DataFrame:
    """Create empirical intervals from individual tree predictions.

    These intervals are useful uncertainty heuristics, but they are not
    guaranteed to be statistically calibrated.
    """
    _validate_coverage(coverage)

    prediction_tensor = _tree_prediction_tensor(
        model=model,
        frame=frame,
    )

    lower_probability = (1.0 - coverage) / 2.0
    upper_probability = 1.0 - lower_probability

    mean_predictions = np.mean(
        prediction_tensor,
        axis=0,
    )
    standard_deviations = np.std(
        prediction_tensor,
        axis=0,
        ddof=0,
    )
    lower_bounds = np.quantile(
        prediction_tensor,
        lower_probability,
        axis=0,
    )
    upper_bounds = np.quantile(
        prediction_tensor,
        upper_probability,
        axis=0,
    )

    output = pd.DataFrame(index=frame.index)

    for target_index, target in enumerate(model.targets):
        output[f"prediction_{target}"] = mean_predictions[
            :,
            target_index,
        ]
        output[f"uncertainty_std_{target}"] = standard_deviations[:, target_index]
        output[f"lower_{target}"] = lower_bounds[
            :,
            target_index,
        ]
        output[f"upper_{target}"] = upper_bounds[
            :,
            target_index,
        ]

    return output


def load_random_forest(path: Path) -> TreeSurrogate:
    """Load and validate a Random-Forest surrogate artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Random-Forest model does not exist: {path}")

    model = TreeSurrogate.load(path)

    if model.model_type != RANDOM_FOREST:
        raise ValueError(f"Expected Random Forest model, found: {model.model_type}")

    return model

"""Tests for ONNX Runtime inference."""

from typing import Any

import numpy as np
import pandas as pd

from edgegenbench.deployment.onnx_inference import (
    OnnxFeasibilityClassifier,
    OnnxSurrogate,
)


def test_onnx_surrogate_matches_sklearn(
    edge_bundle: dict[str, Any],
) -> None:
    """ONNX surrogate predictions should match Scikit-learn."""
    frame: pd.DataFrame = edge_bundle["frame"].iloc[:24]

    sklearn_model = edge_bundle["surrogate"]
    export = edge_bundle["export"]

    onnx_model = OnnxSurrogate.load(
        export.surrogate_onnx_path,
        export.metadata_path,
    )

    sklearn_predictions = sklearn_model.predict(frame)
    onnx_predictions = onnx_model.predict(frame)

    np.testing.assert_allclose(
        onnx_predictions.to_numpy(),
        sklearn_predictions.to_numpy(),
        rtol=1.0e-5,
        atol=1.0e-3,
    )


def test_onnx_classifier_matches_sklearn(
    edge_bundle: dict[str, Any],
) -> None:
    """ONNX classifier probabilities should match."""
    frame: pd.DataFrame = edge_bundle["frame"].iloc[:24]

    sklearn_model = edge_bundle["classifier"]
    export = edge_bundle["export"]

    onnx_model = OnnxFeasibilityClassifier.load(
        export.feasibility_onnx_path,
        export.metadata_path,
    )

    sklearn_probability = sklearn_model.predict_feasibility_probability(frame).to_numpy()

    onnx_probability = onnx_model.predict_feasibility_probability(frame).to_numpy()

    np.testing.assert_allclose(
        onnx_probability,
        sklearn_probability,
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    assert np.array_equal(
        onnx_model.predict(frame).to_numpy(),
        sklearn_model.predict(frame).to_numpy(),
    )

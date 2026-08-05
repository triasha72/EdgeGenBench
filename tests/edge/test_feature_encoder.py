"""Tests for deterministic edge feature encoding."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from edgegenbench.deployment.feature_encoder import (
    EdgeFeatureEncoder,
)


def test_encoder_matches_training_preprocessor(
    edge_bundle: dict[str, Any],
) -> None:
    """The NumPy encoder should reproduce Scikit-learn."""
    frame: pd.DataFrame = edge_bundle["frame"].iloc[:20]

    surrogate = edge_bundle["surrogate"]

    preprocessor = surrogate.pipeline.named_steps["preprocessor"]

    encoder = EdgeFeatureEncoder.from_fitted_preprocessor(preprocessor)

    expected = np.asarray(
        preprocessor.transform(frame),
        dtype=np.float32,
    )

    actual = encoder.transform(frame)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_encoder_rejects_unknown_architecture(
    edge_bundle: dict[str, Any],
) -> None:
    """Unseen architectures should fail clearly."""
    frame: pd.DataFrame = edge_bundle["frame"].iloc[:2].copy()

    surrogate = edge_bundle["surrogate"]

    encoder = EdgeFeatureEncoder.from_fitted_preprocessor(
        surrogate.pipeline.named_steps["preprocessor"]
    )

    frame.loc[
        frame.index[0],
        "propulsion_architecture",
    ] = "unknown_architecture"

    with pytest.raises(
        ValueError,
        match="Unknown propulsion architectures",
    ):
        encoder.transform(frame)

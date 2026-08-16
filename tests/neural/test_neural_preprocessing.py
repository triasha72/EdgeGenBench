"""Tests for neural-surrogate preprocessing."""

import numpy as np
import pandas as pd
import pytest

from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    DEFAULT_TARGETS,
    NUMERIC_FEATURES,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)


def _make_frame() -> pd.DataFrame:
    rows = 8

    data: dict[str, object] = {
        "passenger_capacity": np.linspace(40, 100, rows),
        "design_range_km": np.linspace(500, 2000, rows),
        "cruise_speed_kmh": np.linspace(400, 600, rows),
        "battery_specific_energy_wh_per_kg": np.linspace(
            300,
            600,
            rows,
        ),
        "hydrogen_storage_efficiency": np.linspace(
            0.5,
            0.8,
            rows,
        ),
        "hybridization_ratio": np.linspace(
            0.0,
            1.0,
            rows,
        ),
        CATEGORICAL_FEATURE: [
            "conventional_turboprop",
            "parallel_hybrid",
            "series_hybrid",
            "fuel_cell_electric",
        ]
        * 2,
    }

    for index, target in enumerate(DEFAULT_TARGETS):
        data[target] = np.linspace(
            10.0 + index,
            20.0 + index,
            rows,
        )

    return pd.DataFrame(data)


def test_feature_transform_shape() -> None:
    frame = _make_frame()

    preprocessor = NeuralPreprocessor.fit(frame)

    transformed = preprocessor.transform_features(frame)

    expected_width = len(NUMERIC_FEATURES) + len(preprocessor.categories)

    assert transformed.shape == (
        len(frame),
        expected_width,
    )

    assert transformed.dtype == np.float32


def test_target_round_trip() -> None:
    frame = _make_frame()

    preprocessor = NeuralPreprocessor.fit(frame)

    normalized = preprocessor.transform_targets(frame)

    restored = preprocessor.inverse_transform_targets(normalized)

    expected = frame.loc[
        :,
        list(DEFAULT_TARGETS),
    ].to_numpy(dtype=np.float32)

    np.testing.assert_allclose(
        restored,
        expected,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_transform_does_not_refit_statistics() -> None:
    training_frame = _make_frame()

    preprocessor = NeuralPreprocessor.fit(training_frame)

    original_mean = preprocessor.feature_mean.copy()
    original_scale = preprocessor.feature_scale.copy()

    shifted_frame = training_frame.copy()

    shifted_frame.loc[
        :,
        list(NUMERIC_FEATURES),
    ] += 10000.0

    preprocessor.transform_features(shifted_frame)

    np.testing.assert_array_equal(
        preprocessor.feature_mean,
        original_mean,
    )

    np.testing.assert_array_equal(
        preprocessor.feature_scale,
        original_scale,
    )


def test_unknown_category_is_rejected() -> None:
    frame = _make_frame()

    preprocessor = NeuralPreprocessor.fit(frame)

    invalid_frame = frame.iloc[:1].copy()

    invalid_frame[CATEGORICAL_FEATURE] = "unknown_architecture"

    with pytest.raises(ValueError):
        preprocessor.transform_features(invalid_frame)


def test_preprocessor_save_load_parity(
    tmp_path,
) -> None:
    frame = _make_frame()

    original = NeuralPreprocessor.fit(frame)

    artifact_path = tmp_path / "preprocessing.npz"

    original.save(artifact_path)

    restored = NeuralPreprocessor.load(artifact_path)

    original_features = original.transform_features(frame)

    restored_features = restored.transform_features(frame)

    original_targets = original.transform_targets(frame)

    restored_targets = restored.transform_targets(frame)

    np.testing.assert_array_equal(
        restored_features,
        original_features,
    )

    np.testing.assert_array_equal(
        restored_targets,
        original_targets,
    )

    assert restored.categories == original.categories
    assert restored.targets == original.targets

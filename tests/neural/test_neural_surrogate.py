"""Tests for the compact neural surrogate."""

import pytest
import torch

from edgegenbench.models.neural_surrogate import (
    NeuralSurrogate,
    NeuralSurrogateConfig,
    count_trainable_parameters,
)


def test_neural_surrogate_output_shape() -> None:
    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
    )

    model = NeuralSurrogate(config)

    features = torch.randn(
        8,
        10,
        dtype=torch.float32,
    )

    predictions = model(features)

    assert predictions.shape == (8, 6)


def test_neural_surrogate_batch_one() -> None:
    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
    )

    model = NeuralSurrogate(config)

    features = torch.randn(
        1,
        10,
        dtype=torch.float32,
    )

    predictions = model(features)

    assert predictions.shape == (1, 6)


def test_neural_surrogate_has_trainable_parameters() -> None:
    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
    )

    model = NeuralSurrogate(config)

    assert count_trainable_parameters(model) > 0


def test_neural_surrogate_rejects_wrong_feature_count() -> None:
    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
    )

    model = NeuralSurrogate(config)

    features = torch.randn(
        4,
        9,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        model(features)


def test_config_rejects_invalid_input_dimension() -> None:
    with pytest.raises(ValueError):
        NeuralSurrogateConfig(
            input_dim=0,
            output_dim=6,
        )


def test_config_rejects_invalid_hidden_dimension() -> None:
    with pytest.raises(ValueError):
        NeuralSurrogateConfig(
            input_dim=10,
            output_dim=6,
            hidden_dims=(64, 0, 16),
        )

"""Tests for the compact neural surrogate."""

from pathlib import Path

import pytest
import torch

from edgegenbench.models.neural_surrogate import (
    NeuralSurrogate,
    NeuralSurrogateConfig,
    count_trainable_parameters,
    load_neural_surrogate_checkpoint,
)

TARGETS = (
    "estimated_takeoff_mass_kg",
    "mission_energy_kwh",
    "energy_per_passenger_km_kwh",
    "lifecycle_emissions_proxy_kgco2e",
    "operating_cost_proxy_usd",
    "noise_proxy_db",
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


def test_compact_reference_architecture_parameter_count() -> None:
    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
        hidden_dims=(64, 32, 16),
    )

    model = NeuralSurrogate(config)

    assert count_trainable_parameters(model) == 3414


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


def _write_checkpoint(
    path: Path,
) -> NeuralSurrogate:
    """Create a valid compact-model checkpoint for testing."""
    torch.manual_seed(42)

    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
        hidden_dims=(64, 32, 16),
    )

    model = NeuralSurrogate(config)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dims": list(config.hidden_dims),
            "targets": list(TARGETS),
        },
        path,
    )

    return model


def test_load_neural_surrogate_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"

    reference_model = _write_checkpoint(checkpoint_path)

    loaded_model, targets = load_neural_surrogate_checkpoint(checkpoint_path)

    assert loaded_model.config.input_dim == 10
    assert loaded_model.config.output_dim == 6
    assert loaded_model.config.hidden_dims == (
        64,
        32,
        16,
    )

    assert targets == TARGETS

    assert count_trainable_parameters(loaded_model) == 3414

    features = torch.randn(
        8,
        10,
        dtype=torch.float32,
    )

    reference_model.eval()

    with torch.no_grad():
        reference_predictions = reference_model(features)

        loaded_predictions = loaded_model(features)

    torch.testing.assert_close(
        loaded_predictions,
        reference_predictions,
    )


def test_checkpoint_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(
        FileNotFoundError,
        match="Neural checkpoint does not exist",
    ):
        load_neural_surrogate_checkpoint(missing_path)


def test_checkpoint_loader_rejects_missing_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "invalid.pt"

    torch.save(
        {
            "state_dict": {},
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match="missing required keys",
    ):
        load_neural_surrogate_checkpoint(checkpoint_path)


def test_checkpoint_loader_rejects_target_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "invalid_targets.pt"

    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
    )

    model = NeuralSurrogate(config)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": 10,
            "output_dim": 6,
            "hidden_dims": [64, 32, 16],
            "targets": ["only_one_target"],
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match="target count",
    ):
        load_neural_surrogate_checkpoint(checkpoint_path)

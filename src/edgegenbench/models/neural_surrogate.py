"""Compact PyTorch surrogate model for EdgeGenBench."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class NeuralSurrogateConfig:
    """Architecture configuration for the compact neural surrogate."""

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = (64, 32, 16)

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive.")

        if self.output_dim < 1:
            raise ValueError("output_dim must be positive.")

        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty.")

        if any(width < 1 for width in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive.")


class NeuralSurrogate(nn.Module):
    """Compact feed-forward surrogate for multi-output regression."""

    def __init__(self, config: NeuralSurrogateConfig) -> None:
        super().__init__()

        self.config = config

        layer_dims: Sequence[int] = (
            config.input_dim,
            *config.hidden_dims,
            config.output_dim,
        )

        layers: list[nn.Module] = []

        for index in range(len(layer_dims) - 2):
            layers.append(
                nn.Linear(
                    layer_dims[index],
                    layer_dims[index + 1],
                )
            )
            layers.append(nn.ReLU())

        layers.append(
            nn.Linear(
                layer_dims[-2],
                layer_dims[-1],
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the surrogate."""
        if features.ndim != 2:
            raise ValueError("Expected features with shape [batch_size, feature_count].")

        if features.shape[1] != self.config.input_dim:
            raise ValueError("Input feature dimension does not match model configuration.")

        return self.network(features)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_model_state(
    model: NeuralSurrogate,
    output_path: Path,
) -> None:
    """Save model weights to disk."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        output_path,
    )


def load_neural_surrogate_checkpoint(
    checkpoint_path: Path,
) -> tuple[NeuralSurrogate, tuple[str, ...]]:
    """Reconstruct a trained neural surrogate from its checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Neural checkpoint does not exist: {checkpoint_path}")

    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Neural checkpoint must contain a dictionary.")

    required_keys = {
        "state_dict",
        "input_dim",
        "output_dim",
        "hidden_dims",
        "targets",
    }

    missing_keys = sorted(required_keys.difference(checkpoint))

    if missing_keys:
        raise ValueError(f"Neural checkpoint is missing required keys: {missing_keys}")

    state_dict = checkpoint["state_dict"]

    if not isinstance(state_dict, dict):
        raise ValueError("Neural checkpoint state_dict must be a mapping.")

    try:
        input_dim = int(checkpoint["input_dim"])
        output_dim = int(checkpoint["output_dim"])

        hidden_dims = tuple(int(width) for width in checkpoint["hidden_dims"])

        targets = tuple(str(target) for target in checkpoint["targets"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Neural checkpoint architecture metadata is invalid.") from exc

    if len(targets) != output_dim:
        raise ValueError("Neural checkpoint target count does not match output_dim.")

    config = NeuralSurrogateConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
    )

    model = NeuralSurrogate(config)

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise ValueError("Neural checkpoint weights do not match the stored architecture.") from exc

    model.eval()

    return (
        model,
        targets,
    )

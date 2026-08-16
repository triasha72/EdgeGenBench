"""Tests for the PyTorch neural-surrogate training pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    DEFAULT_TARGETS,
)
from edgegenbench.training.neural_surrogate import (
    train_neural_surrogate,
)


def _write_dataset(
    path: Path,
) -> None:
    rows = 48

    split = ["train"] * 28 + ["validation"] * 10 + ["test"] * 10

    architectures = [
        "conventional_turboprop",
        "parallel_hybrid",
        "series_hybrid",
        "fuel_cell_electric",
    ] * 12

    x = np.linspace(
        0.0,
        1.0,
        rows,
    )

    frame = pd.DataFrame(
        {
            "passenger_capacity": (40.0 + 60.0 * x),
            "design_range_km": (500.0 + 1500.0 * x),
            "cruise_speed_kmh": (400.0 + 200.0 * x),
            "battery_specific_energy_wh_per_kg": (300.0 + 300.0 * x),
            "hydrogen_storage_efficiency": (0.5 + 0.3 * x),
            "hybridization_ratio": x,
            CATEGORICAL_FEATURE: (architectures),
            "split": split,
        }
    )

    for index, target in enumerate(DEFAULT_TARGETS):
        frame[target] = (index + 1) * (10.0 + 3.0 * x)

    frame.to_csv(
        path,
        index=False,
    )


def _write_config(
    path: Path,
) -> None:
    config = {
        "version": "0.2-test",
        "seed": 7,
        "model": {
            "hidden_dims": [
                16,
                8,
            ],
            "activation": "relu",
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "max_epochs": 20,
            "early_stopping_patience": 5,
            "min_delta": 1.0e-6,
        },
        "normalization": {
            "normalize_numeric_features": True,
            "normalize_targets": True,
            "fit_on_split": "train",
        },
        "device": {"preference": ["cpu"]},
    }

    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_neural_training_pipeline(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.csv"

    config_path = tmp_path / "config.yaml"

    output_dir = tmp_path / "artifacts"

    _write_dataset(dataset_path)

    _write_config(config_path)

    artifacts = train_neural_surrogate(
        dataset_path=dataset_path,
        config_path=config_path,
        output_dir=output_dir,
    )

    assert artifacts.model_path.exists()

    assert artifacts.preprocessing_path.exists()

    assert artifacts.training_history_path.exists()

    assert artifacts.test_metrics_path.exists()

    assert artifacts.test_predictions_path.exists()

    assert artifacts.latency_path.exists()

    assert artifacts.summary_path.exists()

    assert artifacts.best_epoch >= 1

    assert artifacts.best_validation_loss >= 0.0

    assert artifacts.parameter_count > 0

    assert artifacts.device == "cpu"


def test_training_uses_expected_test_rows(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.csv"

    config_path = tmp_path / "config.yaml"

    output_dir = tmp_path / "artifacts"

    _write_dataset(dataset_path)

    _write_config(config_path)

    artifacts = train_neural_surrogate(
        dataset_path=dataset_path,
        config_path=config_path,
        output_dir=output_dir,
    )

    predictions = pd.read_csv(artifacts.test_predictions_path)

    assert len(predictions) == 10

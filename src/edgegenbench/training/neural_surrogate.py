"""Training and evaluation pipeline for the PyTorch surrogate."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edgegenbench.evaluation.regression import (
    calculate_regression_metrics,
)
from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    DEFAULT_TARGETS,
    NUMERIC_FEATURES,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)
from edgegenbench.models.neural_surrogate import (
    NeuralSurrogate,
    NeuralSurrogateConfig,
    count_trainable_parameters,
)


@dataclass(frozen=True)
class NeuralTrainingArtifacts:
    """Artifacts produced by neural-surrogate training."""

    model_path: Path
    preprocessing_path: Path
    training_history_path: Path
    test_metrics_path: Path
    test_predictions_path: Path
    latency_path: Path
    summary_path: Path
    best_epoch: int
    best_validation_loss: float
    mean_test_nrmse_std: float
    mean_test_r2: float
    parameter_count: int
    device: str


def _load_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load and validate the neural-training configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Neural config does not exist: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Neural configuration must be a mapping.")

    required_sections = {
        "model",
        "training",
        "device",
    }

    missing = sorted(required_sections.difference(config))

    if missing:
        raise ValueError(f"Neural configuration is missing sections: {missing}")

    return config


def _load_dataset(
    dataset_path: Path,
) -> pd.DataFrame:
    """Load and validate the EdgeGenBench training dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    required_columns = {
        *NUMERIC_FEATURES,
        CATEGORICAL_FEATURE,
        *DEFAULT_TARGETS,
        "split",
    }

    missing_columns = sorted(required_columns.difference(frame.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    available_splits = set(frame["split"].astype(str))

    required_splits = {
        "train",
        "validation",
        "test",
    }

    missing_splits = sorted(required_splits.difference(available_splits))

    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")

    return frame


def _set_seed(seed: int) -> None:
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_device(
    preferences: list[str],
) -> torch.device:
    """Select the first available requested training device."""
    for preference in preferences:
        normalized = str(preference).lower()

        if normalized == "mps":
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                return torch.device("mps")

        elif normalized == "cpu":
            return torch.device("cpu")

        elif normalized == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")

        else:
            raise ValueError(f"Unknown device preference: {preference}")

    return torch.device("cpu")


def _create_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader."""
    feature_tensor = torch.from_numpy(features.astype(np.float32))

    target_tensor = torch.from_numpy(targets.astype(np.float32))

    dataset = TensorDataset(
        feature_tensor,
        target_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _mean_loss(
    model: NeuralSurrogate,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate mean loss without parameter updates."""
    model.eval()

    total_loss = 0.0
    total_rows = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)

            loss = criterion(
                predictions,
                targets,
            )

            batch_rows = int(features.shape[0])

            total_loss += float(loss.item()) * batch_rows

            total_rows += batch_rows

    if total_rows == 0:
        raise ValueError("Evaluation loader contains no rows.")

    return total_loss / total_rows


def _predict(
    model: NeuralSurrogate,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run neural inference and return CPU FP32 outputs."""
    model.eval()

    tensor = torch.from_numpy(features.astype(np.float32)).to(device)

    with torch.no_grad():
        predictions = model(tensor)

    return predictions.detach().cpu().numpy().astype(np.float32)


def _synchronize_device(
    device: torch.device,
) -> None:
    """Synchronize asynchronous accelerator work."""
    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


def _benchmark_latency(
    model: NeuralSurrogate,
    transformed_test_features: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Measure PyTorch inference latency."""
    records: list[dict[str, float | int | str]] = []

    for batch_size in (
        1,
        32,
        256,
    ):
        if len(transformed_test_features) < batch_size:
            continue

        batch = torch.from_numpy(transformed_test_features[:batch_size].astype(np.float32)).to(
            device
        )

        model.eval()

        repeats = 200 if batch_size == 1 else 100

        with torch.no_grad():
            for _ in range(10):
                model(batch)

            _synchronize_device(device)

            elapsed_ms: list[float] = []

            for _ in range(repeats):
                _synchronize_device(device)

                start = perf_counter()

                model(batch)

                _synchronize_device(device)

                elapsed_ms.append((perf_counter() - start) * 1000.0)

        mean_latency = float(np.mean(elapsed_ms))

        p95_latency = float(
            np.percentile(
                elapsed_ms,
                95,
            )
        )

        records.append(
            {
                "runtime": "pytorch",
                "device": str(device),
                "batch_size": batch_size,
                "repeats": repeats,
                "mean_batch_latency_ms": mean_latency,
                "p95_batch_latency_ms": p95_latency,
                "mean_sample_latency_us": (mean_latency * 1000.0 / batch_size),
            }
        )

    return pd.DataFrame(records)


def train_neural_surrogate(
    dataset_path: Path,
    config_path: Path,
    output_dir: Path = Path("artifacts/neural_surrogate"),
) -> NeuralTrainingArtifacts:
    """Train, validate, test, benchmark, and save the PyTorch surrogate."""
    config = _load_config(config_path)

    training_config = config["training"]

    model_config = config["model"]

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    batch_size = int(training_config["batch_size"])

    learning_rate = float(training_config["learning_rate"])

    weight_decay = float(training_config["weight_decay"])

    max_epochs = int(training_config["max_epochs"])

    patience = int(training_config["early_stopping_patience"])

    min_delta = float(training_config["min_delta"])

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    if max_epochs < 1:
        raise ValueError("max_epochs must be positive.")

    if patience < 1:
        raise ValueError("early_stopping_patience must be positive.")

    _set_seed(seed)

    frame = _load_dataset(dataset_path)

    training_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)

    validation_frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)

    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)

    preprocessor = NeuralPreprocessor.fit(training_frame)

    train_features = preprocessor.transform_features(training_frame)

    validation_features = preprocessor.transform_features(validation_frame)

    test_features = preprocessor.transform_features(test_frame)

    train_targets = preprocessor.transform_targets(training_frame)

    validation_targets = preprocessor.transform_targets(validation_frame)

    train_loader = _create_loader(
        features=train_features,
        targets=train_targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    validation_loader = _create_loader(
        features=validation_features,
        targets=validation_targets,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    hidden_dims = tuple(int(width) for width in model_config["hidden_dims"])

    architecture = NeuralSurrogateConfig(
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
        hidden_dims=hidden_dims,
    )

    model = NeuralSurrogate(architecture)

    preferences = [
        str(value)
        for value in config["device"].get(
            "preference",
            ["cpu"],
        )
    ]

    device = _select_device(preferences)

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    best_state: (
        dict[
            str,
            torch.Tensor,
        ]
        | None
    ) = None

    history_records: list[dict[str, float | int]] = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        total_training_loss = 0.0
        total_training_rows = 0

        for features, targets in train_loader:
            features = features.to(device)

            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(features)

            loss = criterion(
                predictions,
                targets,
            )

            loss.backward()

            optimizer.step()

            batch_rows = int(features.shape[0])

            total_training_loss += float(loss.item()) * batch_rows

            total_training_rows += batch_rows

        mean_training_loss = total_training_loss / total_training_rows

        validation_loss = _mean_loss(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history_records.append(
            {
                "epoch": epoch,
                "training_loss": (mean_training_loss),
                "validation_loss": (validation_loss),
            }
        )

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(deepcopy(best_state))

    model = model.to(device)

    normalized_test_predictions = _predict(
        model=model,
        features=test_features,
        device=device,
    )

    physical_test_predictions = preprocessor.inverse_transform_targets(normalized_test_predictions)

    prediction_frame = pd.DataFrame(
        physical_test_predictions,
        columns=DEFAULT_TARGETS,
    )

    test_metrics = calculate_regression_metrics(
        actual=test_frame,
        predicted=prediction_frame,
        targets=DEFAULT_TARGETS,
    )

    mean_test_nrmse_std = float(np.nanmean(test_metrics["nrmse_std"]))

    mean_test_r2 = float(np.nanmean(test_metrics["r2"]))

    report_columns = [
        *NUMERIC_FEATURES,
        CATEGORICAL_FEATURE,
        *DEFAULT_TARGETS,
    ]

    test_prediction_report = (
        test_frame.loc[
            :,
            report_columns,
        ]
        .reset_index(drop=True)
        .copy()
    )

    for target in DEFAULT_TARGETS:
        test_prediction_report[f"predicted_{target}"] = prediction_frame[target].to_numpy()

    latency_report = _benchmark_latency(
        model=model,
        transformed_test_features=(test_features),
        device=device,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / "model.pt"

    preprocessing_path = output_dir / "preprocessing.npz"

    training_history_path = output_dir / "training_history.csv"

    test_metrics_path = output_dir / "test_metrics.csv"

    test_predictions_path = output_dir / "test_predictions.csv"

    latency_path = output_dir / "latency.csv"

    summary_path = output_dir / "summary.json"

    torch.save(
        {
            "state_dict": best_state,
            "input_dim": (architecture.input_dim),
            "output_dim": (architecture.output_dim),
            "hidden_dims": list(architecture.hidden_dims),
            "targets": list(DEFAULT_TARGETS),
        },
        model_path,
    )

    preprocessor.save(preprocessing_path)

    pd.DataFrame(history_records).to_csv(
        training_history_path,
        index=False,
    )

    test_metrics.to_csv(
        test_metrics_path,
        index=False,
    )

    test_prediction_report.to_csv(
        test_predictions_path,
        index=False,
    )

    latency_report.to_csv(
        latency_path,
        index=False,
    )

    parameter_count = count_trainable_parameters(model)

    summary = {
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "model_type": ("compact PyTorch multi-output MLP"),
        "seed": seed,
        "torch_version": (torch.__version__),
        "device": str(device),
        "hidden_dims": list(architecture.hidden_dims),
        "input_dim": (architecture.input_dim),
        "output_dim": (architecture.output_dim),
        "parameter_count": (parameter_count),
        "training_rows": int(len(training_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "best_epoch": (best_epoch),
        "best_validation_loss": (best_validation_loss),
        "epochs_completed": (len(history_records)),
        "mean_test_nrmse_std": (mean_test_nrmse_std),
        "mean_test_r2": (mean_test_r2),
        "model_size_bytes": int(model_path.stat().st_size),
        "model_path": str(model_path),
        "preprocessing_path": str(preprocessing_path),
        "training_history_path": str(training_history_path),
        "test_metrics_path": str(test_metrics_path),
        "test_predictions_path": str(test_predictions_path),
        "latency_path": str(latency_path),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return NeuralTrainingArtifacts(
        model_path=model_path,
        preprocessing_path=(preprocessing_path),
        training_history_path=(training_history_path),
        test_metrics_path=(test_metrics_path),
        test_predictions_path=(test_predictions_path),
        latency_path=latency_path,
        summary_path=summary_path,
        best_epoch=best_epoch,
        best_validation_loss=(best_validation_loss),
        mean_test_nrmse_std=(mean_test_nrmse_std),
        mean_test_r2=(mean_test_r2),
        parameter_count=(parameter_count),
        device=str(device),
    )

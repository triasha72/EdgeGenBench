"""Tests for neural deployment benchmarking."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from edgegenbench.deployment.neural_benchmark import (
    benchmark_neural_onnx,
)
from edgegenbench.deployment.neural_onnx_export import (
    export_neural_surrogate_onnx,
)
from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    DEFAULT_TARGETS,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)
from edgegenbench.models.neural_surrogate import (
    NeuralSurrogate,
    NeuralSurrogateConfig,
)


def _make_dataset(
    rows: int = 40,
) -> pd.DataFrame:
    """Create a small benchmark dataset."""
    architectures = (
        "conventional_turboprop",
        "parallel_hybrid",
        "series_hybrid",
        "fuel_cell_electric",
    )

    data: dict[str, object] = {
        "passenger_capacity": np.linspace(
            40,
            100,
            rows,
        ),
        "design_range_km": np.linspace(
            500,
            2000,
            rows,
        ),
        "cruise_speed_kmh": np.linspace(
            400,
            600,
            rows,
        ),
        ("battery_specific_energy_wh_per_kg"): np.linspace(
            300,
            600,
            rows,
        ),
        "hydrogen_storage_efficiency": (
            np.linspace(
                0.5,
                0.8,
                rows,
            )
        ),
        "hybridization_ratio": np.linspace(
            0.0,
            1.0,
            rows,
        ),
        CATEGORICAL_FEATURE: [architectures[index % len(architectures)] for index in range(rows)],
        "split": ["train" if index < 30 else "test" for index in range(rows)],
    }

    for index, target in enumerate(DEFAULT_TARGETS):
        data[target] = np.linspace(
            10.0 + index,
            30.0 + index,
            rows,
        )

    return pd.DataFrame(data)


def _write_benchmark_artifacts(
    directory: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
]:
    """Create dataset and model artifacts."""
    frame = _make_dataset()

    dataset_path = directory / "dataset.csv"

    frame.to_csv(
        dataset_path,
        index=False,
    )

    training_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)

    preprocessor = NeuralPreprocessor.fit(training_frame)

    config = NeuralSurrogateConfig(
        input_dim=preprocessor.input_dim,
        output_dim=preprocessor.output_dim,
        hidden_dims=(64, 32, 16),
    )

    torch.manual_seed(42)

    model = NeuralSurrogate(config)

    model_path = directory / "model.pt"

    preprocessing_path = directory / "preprocessing.npz"

    torch.save(
        {
            "state_dict": (model.state_dict()),
            "input_dim": (config.input_dim),
            "output_dim": (config.output_dim),
            "hidden_dims": list(config.hidden_dims),
            "targets": list(DEFAULT_TARGETS),
        },
        model_path,
    )

    preprocessor.save(preprocessing_path)

    export_artifacts = export_neural_surrogate_onnx(
        model_path=model_path,
        preprocessing_path=(preprocessing_path),
        output_dir=(directory / "onnx"),
    )

    return (
        dataset_path,
        model_path,
        preprocessing_path,
        export_artifacts.onnx_path,
        export_artifacts.metadata_path,
    )


def test_neural_benchmark_outputs(
    tmp_path: Path,
) -> None:
    (
        dataset_path,
        model_path,
        preprocessing_path,
        onnx_path,
        metadata_path,
    ) = _write_benchmark_artifacts(tmp_path)

    artifacts = benchmark_neural_onnx(
        dataset_path=dataset_path,
        model_path=model_path,
        preprocessing_path=(preprocessing_path),
        onnx_model_path=onnx_path,
        metadata_path=metadata_path,
        output_dir=(tmp_path / "benchmark"),
        batch_sizes=(
            1,
            4,
        ),
        repeats=2,
        warmups=1,
    )

    assert artifacts.test_rows == 10

    assert artifacts.normalized_equivalent is True

    assert artifacts.normalized_max_absolute_difference < 1.0e-5

    assert artifacts.equivalence_path.exists()

    assert artifacts.latency_path.exists()

    assert artifacts.summary_path.exists()


def test_neural_benchmark_latency_schema(
    tmp_path: Path,
) -> None:
    (
        dataset_path,
        model_path,
        preprocessing_path,
        onnx_path,
        metadata_path,
    ) = _write_benchmark_artifacts(tmp_path)

    artifacts = benchmark_neural_onnx(
        dataset_path=dataset_path,
        model_path=model_path,
        preprocessing_path=(preprocessing_path),
        onnx_model_path=onnx_path,
        metadata_path=metadata_path,
        output_dir=(tmp_path / "benchmark"),
        batch_sizes=(
            1,
            4,
        ),
        repeats=2,
        warmups=1,
    )

    latency = pd.read_csv(artifacts.latency_path)

    assert set(latency["runtime"]) == {
        "pytorch_cpu",
        "onnxruntime_cpu",
    }

    assert set(latency["batch_size"]) == {
        1,
        4,
    }

    expected_columns = {
        "runtime",
        "batch_size",
        "repeats",
        "warmups",
        "mean_batch_latency_ms",
        "p95_batch_latency_ms",
        "mean_sample_latency_us",
    }

    assert expected_columns.issubset(latency.columns)

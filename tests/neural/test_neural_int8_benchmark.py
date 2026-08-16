"""Tests for mixed-precision INT8 neural ONNX benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch

from edgegenbench.deployment.neural_int8 import (
    export_neural_surrogate_int8,
)
from edgegenbench.deployment.neural_int8_benchmark import (
    _evaluate_drift_limits,
    _load_test_frame,
    _normalize_batch_sizes,
    _summarize_latency_runs,
    benchmark_neural_int8,
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

    data: dict[
        str,
        object,
    ] = {
        "passenger_capacity": (
            np.linspace(
                40,
                100,
                rows,
            )
        ),
        "design_range_km": (
            np.linspace(
                500,
                2000,
                rows,
            )
        ),
        "cruise_speed_kmh": (
            np.linspace(
                400,
                600,
                rows,
            )
        ),
        "battery_specific_energy_wh_per_kg": (
            np.linspace(
                300,
                600,
                rows,
            )
        ),
        "hydrogen_storage_efficiency": (
            np.linspace(
                0.5,
                0.8,
                rows,
            )
        ),
        "hybridization_ratio": (
            np.linspace(
                0.0,
                1.0,
                rows,
            )
        ),
        CATEGORICAL_FEATURE: [architectures[index % len(architectures)] for index in range(rows)],
        "split": [("train" if index < 30 else "test") for index in range(rows)],
    }

    for index, target in enumerate(DEFAULT_TARGETS):
        data[target] = np.linspace(
            10.0 + index,
            30.0 + index,
            rows,
        )

    return pd.DataFrame(data)


def _write_int8_benchmark_artifacts(
    directory: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
]:
    """Create dataset, preprocessing, FP32, and mixed-INT8 artifacts."""
    frame = _make_dataset()

    dataset_path = directory / "dataset.csv"

    frame.to_csv(
        dataset_path,
        index=False,
    )

    training_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)

    preprocessor = NeuralPreprocessor.fit(training_frame)

    preprocessing_path = directory / "preprocessing.npz"

    preprocessor.save(preprocessing_path)

    config = NeuralSurrogateConfig(
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
        hidden_dims=(
            64,
            32,
            16,
        ),
    )

    torch.manual_seed(42)

    model = NeuralSurrogate(config)

    model_path = directory / "model.pt"

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

    fp32 = export_neural_surrogate_onnx(
        model_path=(model_path),
        preprocessing_path=(preprocessing_path),
        output_dir=(directory / "fp32"),
    )

    int8 = export_neural_surrogate_int8(
        fp32_model_path=(fp32.onnx_path),
        dataset_path=(dataset_path),
        preprocessing_path=(preprocessing_path),
        output_dir=(directory / "int8"),
        calibration_batch_size=8,
    )

    return (
        dataset_path,
        preprocessing_path,
        fp32.onnx_path,
        int8.onnx_path,
    )


def test_normalize_batch_sizes() -> None:
    """Batch-size normalization removes duplicates and invalid values."""
    assert _normalize_batch_sizes(
        (
            32,
            1,
            32,
            0,
            -1,
            4,
        )
    ) == (
        1,
        4,
        32,
    )


def test_normalize_batch_sizes_rejects_empty() -> None:
    """At least one valid batch size is required."""
    with pytest.raises(
        ValueError,
        match=("At least one positive batch size"),
    ):
        _normalize_batch_sizes(
            (
                0,
                -1,
            )
        )


def test_evaluate_drift_limits() -> None:
    """All four INT8 drift guards are evaluated independently."""
    assert _evaluate_drift_limits(
        mean_drift=0.008,
        p99_drift=0.025,
        p999_drift=0.040,
        max_drift=0.060,
        max_mean_normalized_drift=0.015,
        max_p99_normalized_drift=0.040,
        max_p999_normalized_drift=0.060,
        max_normalized_drift=0.080,
    ) == (
        True,
        True,
        True,
        True,
    )

    assert _evaluate_drift_limits(
        mean_drift=0.020,
        p99_drift=0.025,
        p999_drift=0.040,
        max_drift=0.060,
        max_mean_normalized_drift=0.015,
        max_p99_normalized_drift=0.040,
        max_p999_normalized_drift=0.060,
        max_normalized_drift=0.080,
    ) == (
        False,
        True,
        True,
        True,
    )


@pytest.mark.parametrize(
    (
        "keyword",
        "value",
        "message",
    ),
    [
        (
            "runs",
            0,
            "runs must be at least one",
        ),
        (
            "repeats",
            0,
            "repeats must be at least one",
        ),
        (
            "warmups",
            -1,
            "warmups cannot be negative",
        ),
        (
            "max_mean_normalized_drift",
            -0.1,
            ("max_mean_normalized_drift cannot be negative"),
        ),
        (
            "max_p99_normalized_drift",
            -0.1,
            ("max_p99_normalized_drift cannot be negative"),
        ),
        (
            "max_p999_normalized_drift",
            -0.1,
            ("max_p999_normalized_drift cannot be negative"),
        ),
        (
            "max_normalized_drift",
            -0.1,
            ("max_normalized_drift cannot be negative"),
        ),
    ],
)
def test_benchmark_configuration_validation(
    keyword: str,
    value: int | float,
    message: str,
) -> None:
    """Invalid benchmark configuration fails before artifact loading."""
    kwargs: dict[
        str,
        int | float,
    ] = {
        "runs": 1,
        "repeats": 1,
        "warmups": 0,
        "max_mean_normalized_drift": (0.015),
        "max_p99_normalized_drift": (0.040),
        "max_p999_normalized_drift": (0.060),
        "max_normalized_drift": (0.080),
    }

    kwargs[keyword] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        benchmark_neural_int8(
            dataset_path=Path("missing.csv"),
            preprocessing_path=Path("missing.npz"),
            fp32_model_path=Path("missing_fp32.onnx"),
            int8_model_path=Path("missing_int8.onnx"),
            runs=int(kwargs["runs"]),
            repeats=int(kwargs["repeats"]),
            warmups=int(kwargs["warmups"]),
            max_mean_normalized_drift=float(kwargs["max_mean_normalized_drift"]),
            max_p99_normalized_drift=float(kwargs["max_p99_normalized_drift"]),
            max_p999_normalized_drift=float(kwargs["max_p999_normalized_drift"]),
            max_normalized_drift=float(kwargs["max_normalized_drift"]),
        )


def test_missing_dataset_is_rejected(
    tmp_path: Path,
) -> None:
    """Missing benchmark dataset produces a clear error."""
    with pytest.raises(
        FileNotFoundError,
        match=("Dataset does not exist"),
    ):
        benchmark_neural_int8(
            dataset_path=(tmp_path / "missing.csv"),
            preprocessing_path=(tmp_path / "missing.npz"),
            fp32_model_path=(tmp_path / "fp32.onnx"),
            int8_model_path=(tmp_path / "int8.onnx"),
            runs=1,
            repeats=1,
            warmups=0,
        )


def test_empty_test_split_is_rejected(
    tmp_path: Path,
) -> None:
    """Dataset without test rows cannot be benchmarked."""
    path = tmp_path / "dataset.csv"

    pd.DataFrame(
        {
            "split": [
                "train",
                "train",
            ]
        }
    ).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=("No rows are available"),
    ):
        _load_test_frame(path)


def test_latency_summary() -> None:
    """Repeated CPU latency measurements aggregate by batch."""
    runs = pd.DataFrame(
        [
            {
                "run": 1,
                "batch_size": 1,
                "fp32_mean_ms": 0.040,
                "fp32_p95_ms": 0.050,
                "int8_mean_ms": 0.050,
                "int8_p95_ms": 0.060,
                "fp32_over_int8": 0.8,
            },
            {
                "run": 2,
                "batch_size": 1,
                "fp32_mean_ms": 0.060,
                "fp32_p95_ms": 0.070,
                "int8_mean_ms": 0.040,
                "int8_p95_ms": 0.050,
                "fp32_over_int8": 1.5,
            },
            {
                "run": 3,
                "batch_size": 1,
                "fp32_mean_ms": 0.050,
                "fp32_p95_ms": 0.060,
                "int8_mean_ms": 0.050,
                "int8_p95_ms": 0.060,
                "fp32_over_int8": 1.0,
            },
        ]
    )

    summary = _summarize_latency_runs(runs)

    assert len(summary) == 1

    row = summary.iloc[0]

    assert int(row["batch_size"]) == 1

    assert float(row["fp32_median_ms"]) == pytest.approx(0.050)

    assert float(row["int8_median_ms"]) == pytest.approx(0.050)

    assert float(row["median_fp32_over_int8_ratio"]) == pytest.approx(1.0)

    assert int(row["int8_faster_runs"]) == 1

    assert int(row["total_runs"]) == 3


@pytest.mark.skipif(
    ("CPUExecutionProvider" not in ort.get_available_providers()),
    reason=("CPUExecutionProvider unavailable"),
)
def test_int8_benchmark_outputs(
    tmp_path: Path,
) -> None:
    """CPU integration writes drift, quality, and latency artifacts."""
    (
        dataset_path,
        preprocessing_path,
        fp32_model_path,
        int8_model_path,
    ) = _write_int8_benchmark_artifacts(tmp_path)

    artifacts = benchmark_neural_int8(
        dataset_path=(dataset_path),
        preprocessing_path=(preprocessing_path),
        fp32_model_path=(fp32_model_path),
        int8_model_path=(int8_model_path),
        output_dir=(tmp_path / "benchmark"),
        batch_sizes=(
            1,
            4,
        ),
        runs=1,
        repeats=2,
        warmups=1,
        max_mean_normalized_drift=10.0,
        max_p99_normalized_drift=10.0,
        max_p999_normalized_drift=10.0,
        max_normalized_drift=10.0,
    )

    assert artifacts.test_rows == 10

    assert artifacts.equivalence_path.exists()

    assert artifacts.task_metrics_path.exists()

    assert artifacts.latency_runs_path.exists()

    assert artifacts.latency_summary_path.exists()

    assert artifacts.summary_path.exists()

    assert np.isfinite(artifacts.int8_mean_nrmse_std)

    assert np.isfinite(artifacts.int8_mean_r2)

    assert artifacts.mean_drift_within_limit

    assert artifacts.p99_drift_within_limit

    assert artifacts.p999_drift_within_limit

    assert artifacts.max_drift_within_limit

    latency_runs = pd.read_csv(artifacts.latency_runs_path)

    latency_summary = pd.read_csv(artifacts.latency_summary_path)

    equivalence = pd.read_csv(artifacts.equivalence_path)

    task_metrics = pd.read_csv(artifacts.task_metrics_path)

    assert len(latency_runs) == 2

    assert set(latency_runs["batch_size"]) == {
        1,
        4,
    }

    assert set(latency_summary["batch_size"]) == {
        1,
        4,
    }

    assert set(equivalence["target"]) == set(DEFAULT_TARGETS)

    assert set(task_metrics["target"]) == set(DEFAULT_TARGETS)

    assert set(task_metrics["precision"]) == {
        "fp32",
        "mixed_int8_fp32",
    }

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))

    assert summary["provider"] == "CPUExecutionProvider"

    assert summary["precision"] == "mixed_int8_fp32"

    assert summary["reference_precision"] == "fp32"

    assert summary["latency"]["runs"] == 1

    assert summary["latency"]["repeats"] == 2

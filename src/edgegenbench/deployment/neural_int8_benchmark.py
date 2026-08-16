"""Accuracy, drift, size, and CPU latency benchmarking for mixed INT8 neural ONNX."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from platform import platform
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from edgegenbench.evaluation.regression import (
    calculate_regression_metrics,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)

DEFAULT_INT8_BENCHMARK_BATCH_SIZES = (
    1,
    32,
    256,
)

DEFAULT_MAX_MEAN_NORMALIZED_DRIFT = 0.015
DEFAULT_MAX_P99_NORMALIZED_DRIFT = 0.040
DEFAULT_MAX_P999_NORMALIZED_DRIFT = 0.060
DEFAULT_MAX_NORMALIZED_DRIFT = 0.080


@dataclass(frozen=True)
class NeuralInt8BenchmarkArtifacts:
    """Artifacts produced by FP32-versus-mixed-INT8 CPU benchmarking."""

    equivalence_path: Path
    task_metrics_path: Path
    latency_runs_path: Path
    latency_summary_path: Path
    summary_path: Path
    test_rows: int
    normalized_mean_absolute_difference: float
    normalized_p95_absolute_difference: float
    normalized_p99_absolute_difference: float
    normalized_p999_absolute_difference: float
    normalized_max_absolute_difference: float
    int8_mean_nrmse_std: float
    int8_mean_r2: float
    mean_drift_within_limit: bool
    p99_drift_within_limit: bool
    p999_drift_within_limit: bool
    max_drift_within_limit: bool


def _normalize_batch_sizes(
    batch_sizes: Sequence[int],
) -> tuple[int, ...]:
    """Return sorted unique positive benchmark batch sizes."""
    normalized = tuple(
        sorted({int(batch_size) for batch_size in batch_sizes if int(batch_size) > 0})
    )

    if not normalized:
        raise ValueError("At least one positive batch size is required.")

    return normalized


def _validate_benchmark_configuration(
    *,
    runs: int,
    repeats: int,
    warmups: int,
    max_mean_normalized_drift: float,
    max_p99_normalized_drift: float,
    max_p999_normalized_drift: float,
    max_normalized_drift: float,
) -> None:
    """Validate iteration counts and INT8 drift guardrails."""
    if runs < 1:
        raise ValueError("runs must be at least one.")

    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    if warmups < 0:
        raise ValueError("warmups cannot be negative.")

    guards = {
        "max_mean_normalized_drift": (max_mean_normalized_drift),
        "max_p99_normalized_drift": (max_p99_normalized_drift),
        "max_p999_normalized_drift": (max_p999_normalized_drift),
        "max_normalized_drift": (max_normalized_drift),
    }

    for name, value in guards.items():
        if value < 0.0:
            raise ValueError(f"{name} cannot be negative.")


def _load_test_frame(
    dataset_path: Path,
) -> pd.DataFrame:
    """Load the explicit held-out test partition."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    if "split" not in frame.columns:
        raise ValueError("INT8 benchmarking requires a dataset with a split column.")

    test_frame = frame.loc[frame["split"].astype(str) == "test"].reset_index(drop=True)

    if test_frame.empty:
        raise ValueError("No rows are available for INT8 benchmarking.")

    return test_frame


def _create_cpu_session(
    model_path: Path,
) -> ort.InferenceSession:
    """Create a CPU-only ONNX Runtime session."""
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CPUExecutionProvider is unavailable.")

    return ort.InferenceSession(
        str(model_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )


def _validate_session_interface(
    session: ort.InferenceSession,
    *,
    input_dim: int,
    output_dim: int,
) -> tuple[str, str]:
    """Validate the neural runtime interface."""
    inputs = session.get_inputs()

    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError("Expected exactly one neural ONNX input.")

    if len(outputs) != 1:
        raise RuntimeError("Expected exactly one neural ONNX output.")

    input_shape = inputs[0].shape

    output_shape = outputs[0].shape

    if len(input_shape) != 2:
        raise RuntimeError("Expected a rank-two neural ONNX input.")

    if len(output_shape) != 2:
        raise RuntimeError("Expected a rank-two neural ONNX output.")

    if int(input_shape[1]) != int(input_dim):
        raise RuntimeError("ONNX input width does not match preprocessing.")

    if int(output_shape[1]) != int(output_dim):
        raise RuntimeError("ONNX output width does not match preprocessing.")

    return (
        inputs[0].name,
        outputs[0].name,
    )


def _predict_session(
    session: ort.InferenceSession,
    features: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
) -> np.ndarray:
    """Run a complete dynamic batch."""
    (
        input_name,
        output_name,
    ) = _validate_session_interface(
        session,
        input_dim=(input_dim),
        output_dim=(output_dim),
    )

    raw = session.run(
        [
            output_name,
        ],
        {
            input_name: features,
        },
    )[0]

    prediction = np.asarray(
        raw,
        dtype=np.float32,
    )

    expected_shape = (
        len(features),
        output_dim,
    )

    if prediction.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected ONNX output shape: {prediction.shape}; expected {expected_shape}."
        )

    if not np.isfinite(prediction).all():
        raise RuntimeError("ONNX prediction contains non-finite values.")

    return prediction


def _measure_latency(
    operation: Callable[
        [],
        object,
    ],
    *,
    repeats: int,
    warmups: int,
) -> tuple[
    float,
    float,
]:
    """Measure mean and P95 latency excluding session construction."""
    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    if warmups < 0:
        raise ValueError("warmups cannot be negative.")

    gc_was_enabled = gc.isenabled()

    gc.disable()

    try:
        for _ in range(warmups):
            operation()

        elapsed_ms: list[float] = []

        for _ in range(repeats):
            start = perf_counter()

            operation()

            elapsed_ms.append((perf_counter() - start) * 1000.0)
    finally:
        if gc_was_enabled:
            gc.enable()

    return (
        float(np.mean(elapsed_ms)),
        float(
            np.percentile(
                elapsed_ms,
                95,
            )
        ),
    )


def _evaluate_drift_limits(
    *,
    mean_drift: float,
    p99_drift: float,
    p999_drift: float,
    max_drift: float,
    max_mean_normalized_drift: float,
    max_p99_normalized_drift: float,
    max_p999_normalized_drift: float,
    max_normalized_drift: float,
) -> tuple[
    bool,
    bool,
    bool,
    bool,
]:
    """Evaluate INT8 drift against validation-derived regression guards."""
    return (
        bool(mean_drift <= max_mean_normalized_drift),
        bool(p99_drift <= max_p99_normalized_drift),
        bool(p999_drift <= max_p999_normalized_drift),
        bool(max_drift <= max_normalized_drift),
    )


def _summarize_latency_runs(
    latency_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate repeated FP32-versus-INT8 CPU latency runs."""
    required_columns = {
        "run",
        "batch_size",
        "fp32_mean_ms",
        "fp32_p95_ms",
        "int8_mean_ms",
        "int8_p95_ms",
        "fp32_over_int8",
    }

    missing = sorted(required_columns.difference(latency_runs.columns))

    if missing:
        raise ValueError(f"Latency runs are missing required columns: {missing}")

    if latency_runs.empty:
        raise ValueError("Latency runs cannot be empty.")

    records: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    batch_values = sorted(int(value) for value in latency_runs["batch_size"].unique())

    for batch_size in batch_values:
        rows = latency_runs.loc[latency_runs["batch_size"] == batch_size]

        ratios = rows["fp32_over_int8"]

        records.append(
            {
                "batch_size": (batch_size),
                "fp32_median_ms": float(rows["fp32_mean_ms"].median()),
                "fp32_median_p95_ms": float(rows["fp32_p95_ms"].median()),
                "int8_median_ms": float(rows["int8_mean_ms"].median()),
                "int8_median_p95_ms": float(rows["int8_p95_ms"].median()),
                "median_fp32_over_int8_ratio": float(ratios.median()),
                "min_fp32_over_int8_ratio": float(ratios.min()),
                "max_fp32_over_int8_ratio": float(ratios.max()),
                "int8_faster_runs": int((ratios > 1.0).sum()),
                "total_runs": int(len(rows)),
            }
        )

    return pd.DataFrame(records)


def _build_equivalence_table(
    *,
    targets: Sequence[str],
    preprocessor: NeuralPreprocessor,
    fp32_normalized: np.ndarray,
    int8_normalized: np.ndarray,
) -> pd.DataFrame:
    """Build target-level INT8 drift statistics."""
    fp32_physical = preprocessor.inverse_transform_targets(fp32_normalized)

    int8_physical = preprocessor.inverse_transform_targets(int8_normalized)

    records: list[
        dict[
            str,
            float | str,
        ]
    ] = []

    for index, target in enumerate(targets):
        normalized_difference = np.abs(
            fp32_normalized[
                :,
                index,
            ]
            - int8_normalized[
                :,
                index,
            ]
        )

        physical_difference = np.abs(
            fp32_physical[
                :,
                index,
            ]
            - int8_physical[
                :,
                index,
            ]
        )

        reference_scale = float(
            np.max(
                np.abs(
                    fp32_physical[
                        :,
                        index,
                    ]
                )
            )
        )

        physical_max = float(np.max(physical_difference))

        relative_max = physical_max / reference_scale if reference_scale > 0.0 else 0.0

        records.append(
            {
                "target": str(target),
                "normalized_mean_abs": float(np.mean(normalized_difference)),
                "normalized_p95_abs": float(
                    np.quantile(
                        normalized_difference,
                        0.95,
                    )
                ),
                "normalized_p99_abs": float(
                    np.quantile(
                        normalized_difference,
                        0.99,
                    )
                ),
                "normalized_p999_abs": float(
                    np.quantile(
                        normalized_difference,
                        0.999,
                    )
                ),
                "normalized_max_abs": float(np.max(normalized_difference)),
                "physical_mean_abs": float(np.mean(physical_difference)),
                "physical_max_abs": (physical_max),
                "max_reference_relative": float(relative_max),
            }
        )

    return pd.DataFrame(records)


def _build_task_metrics(
    *,
    test_frame: pd.DataFrame,
    targets: Sequence[str],
    preprocessor: NeuralPreprocessor,
    fp32_normalized: np.ndarray,
    int8_normalized: np.ndarray,
) -> pd.DataFrame:
    """Build FP32 and mixed-INT8 predictive-quality metrics."""
    fp32_physical = preprocessor.inverse_transform_targets(fp32_normalized)

    int8_physical = preprocessor.inverse_transform_targets(int8_normalized)

    fp32_frame = pd.DataFrame(
        fp32_physical,
        columns=targets,
        index=(test_frame.index),
    )

    int8_frame = pd.DataFrame(
        int8_physical,
        columns=targets,
        index=(test_frame.index),
    )

    fp32_metrics = calculate_regression_metrics(
        actual=test_frame,
        predicted=(fp32_frame),
        targets=targets,
    )

    fp32_metrics.insert(
        0,
        "precision",
        "fp32",
    )

    int8_metrics = calculate_regression_metrics(
        actual=test_frame,
        predicted=(int8_frame),
        targets=targets,
    )

    int8_metrics.insert(
        0,
        "precision",
        "mixed_int8_fp32",
    )

    return pd.concat(
        [
            fp32_metrics,
            int8_metrics,
        ],
        ignore_index=True,
    )


def benchmark_neural_int8(
    dataset_path: Path,
    preprocessing_path: Path,
    fp32_model_path: Path,
    int8_model_path: Path,
    output_dir: Path = Path("artifacts/neural_int8_benchmark"),
    batch_sizes: Sequence[int] = (DEFAULT_INT8_BENCHMARK_BATCH_SIZES),
    runs: int = 5,
    repeats: int = 500,
    warmups: int = 50,
    max_mean_normalized_drift: float = (DEFAULT_MAX_MEAN_NORMALIZED_DRIFT),
    max_p99_normalized_drift: float = (DEFAULT_MAX_P99_NORMALIZED_DRIFT),
    max_p999_normalized_drift: float = (DEFAULT_MAX_P999_NORMALIZED_DRIFT),
    max_normalized_drift: float = (DEFAULT_MAX_NORMALIZED_DRIFT),
) -> NeuralInt8BenchmarkArtifacts:
    """Benchmark canonical FP32 and mixed-INT8 ONNX models on CPU."""
    _validate_benchmark_configuration(
        runs=runs,
        repeats=repeats,
        warmups=warmups,
        max_mean_normalized_drift=(max_mean_normalized_drift),
        max_p99_normalized_drift=(max_p99_normalized_drift),
        max_p999_normalized_drift=(max_p999_normalized_drift),
        max_normalized_drift=(max_normalized_drift),
    )

    normalized_batch_sizes = _normalize_batch_sizes(batch_sizes)

    test_frame = _load_test_frame(dataset_path)

    if not (preprocessing_path.exists()):
        raise FileNotFoundError(f"Preprocessing artifact does not exist: {preprocessing_path}")

    preprocessor = NeuralPreprocessor.load(preprocessing_path)

    targets = tuple(preprocessor.targets)

    transformed_features = preprocessor.transform_features(test_frame).astype(
        np.float32,
        copy=False,
    )

    usable_batch_sizes = tuple(
        batch_size
        for batch_size in normalized_batch_sizes
        if batch_size <= len(transformed_features)
    )

    if not usable_batch_sizes:
        raise ValueError("No requested batch size fits the available test rows.")

    fp32_session = _create_cpu_session(fp32_model_path)

    int8_session = _create_cpu_session(int8_model_path)

    fp32_normalized = _predict_session(
        fp32_session,
        transformed_features,
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
    )

    int8_normalized = _predict_session(
        int8_session,
        transformed_features,
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
    )

    difference = np.abs(fp32_normalized - int8_normalized)

    normalized_mean_absolute_difference = float(np.mean(difference))

    normalized_p95_absolute_difference = float(
        np.quantile(
            difference,
            0.95,
        )
    )

    normalized_p99_absolute_difference = float(
        np.quantile(
            difference,
            0.99,
        )
    )

    normalized_p999_absolute_difference = float(
        np.quantile(
            difference,
            0.999,
        )
    )

    normalized_max_absolute_difference = float(np.max(difference))

    (
        mean_drift_within_limit,
        p99_drift_within_limit,
        p999_drift_within_limit,
        max_drift_within_limit,
    ) = _evaluate_drift_limits(
        mean_drift=(normalized_mean_absolute_difference),
        p99_drift=(normalized_p99_absolute_difference),
        p999_drift=(normalized_p999_absolute_difference),
        max_drift=(normalized_max_absolute_difference),
        max_mean_normalized_drift=(max_mean_normalized_drift),
        max_p99_normalized_drift=(max_p99_normalized_drift),
        max_p999_normalized_drift=(max_p999_normalized_drift),
        max_normalized_drift=(max_normalized_drift),
    )

    equivalence = _build_equivalence_table(
        targets=targets,
        preprocessor=(preprocessor),
        fp32_normalized=(fp32_normalized),
        int8_normalized=(int8_normalized),
    )

    task_metrics = _build_task_metrics(
        test_frame=(test_frame),
        targets=targets,
        preprocessor=(preprocessor),
        fp32_normalized=(fp32_normalized),
        int8_normalized=(int8_normalized),
    )

    int8_rows = task_metrics.loc[task_metrics["precision"] == "mixed_int8_fp32"]

    int8_mean_nrmse_std = float(np.nanmean(int8_rows["nrmse_std"].to_numpy(dtype=np.float64)))

    int8_mean_r2 = float(np.nanmean(int8_rows["r2"].to_numpy(dtype=np.float64)))

    latency_records: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    for batch_size in usable_batch_sizes:
        numpy_batch = transformed_features[:batch_size].astype(
            np.float32,
            copy=False,
        )

        (
            fp32_input_name,
            fp32_output_name,
        ) = _validate_session_interface(
            fp32_session,
            input_dim=(preprocessor.input_dim),
            output_dim=(preprocessor.output_dim),
        )

        (
            int8_input_name,
            int8_output_name,
        ) = _validate_session_interface(
            int8_session,
            input_dim=(preprocessor.input_dim),
            output_dim=(preprocessor.output_dim),
        )

        def fp32_operation(
            session: ort.InferenceSession = (fp32_session),
            input_name: str = (fp32_input_name),
            output_name: str = (fp32_output_name),
            batch: np.ndarray = (numpy_batch),
        ) -> object:
            return session.run(
                [
                    output_name,
                ],
                {
                    input_name: batch,
                },
            )

        def int8_operation(
            session: ort.InferenceSession = (int8_session),
            input_name: str = (int8_input_name),
            output_name: str = (int8_output_name),
            batch: np.ndarray = (numpy_batch),
        ) -> object:
            return session.run(
                [
                    output_name,
                ],
                {
                    input_name: batch,
                },
            )

        for run in range(
            1,
            runs + 1,
        ):
            if run % 2 == 1:
                (
                    fp32_mean_ms,
                    fp32_p95_ms,
                ) = _measure_latency(
                    fp32_operation,
                    repeats=(repeats),
                    warmups=(warmups),
                )

                (
                    int8_mean_ms,
                    int8_p95_ms,
                ) = _measure_latency(
                    int8_operation,
                    repeats=(repeats),
                    warmups=(warmups),
                )
            else:
                (
                    int8_mean_ms,
                    int8_p95_ms,
                ) = _measure_latency(
                    int8_operation,
                    repeats=(repeats),
                    warmups=(warmups),
                )

                (
                    fp32_mean_ms,
                    fp32_p95_ms,
                ) = _measure_latency(
                    fp32_operation,
                    repeats=(repeats),
                    warmups=(warmups),
                )

            if int8_mean_ms <= 0.0:
                raise RuntimeError("INT8 latency must be positive.")

            latency_records.append(
                {
                    "run": run,
                    "batch_size": (batch_size),
                    "fp32_mean_ms": (fp32_mean_ms),
                    "fp32_p95_ms": (fp32_p95_ms),
                    "int8_mean_ms": (int8_mean_ms),
                    "int8_p95_ms": (int8_p95_ms),
                    "fp32_over_int8": (fp32_mean_ms / int8_mean_ms),
                }
            )

    latency_runs = pd.DataFrame(latency_records)

    latency_summary = _summarize_latency_runs(latency_runs)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    equivalence_path = output_dir / "equivalence.csv"

    task_metrics_path = output_dir / "task_metrics.csv"

    latency_runs_path = output_dir / "latency_runs.csv"

    latency_summary_path = output_dir / "latency_summary.csv"

    summary_path = output_dir / "summary.json"

    equivalence.to_csv(
        equivalence_path,
        index=False,
    )

    task_metrics.to_csv(
        task_metrics_path,
        index=False,
    )

    latency_runs.to_csv(
        latency_runs_path,
        index=False,
    )

    latency_summary.to_csv(
        latency_summary_path,
        index=False,
    )

    fp32_model_size_bytes = int(fp32_model_path.stat().st_size)

    int8_model_size_bytes = int(int8_model_path.stat().st_size)

    if fp32_model_size_bytes < 1 or int8_model_size_bytes < 1:
        raise RuntimeError("Benchmark model artifacts must be non-empty.")

    size_reduction_percent = (1.0 - (int8_model_size_bytes / fp32_model_size_bytes)) * 100.0

    latency_summary_records = json.loads(latency_summary.to_json(orient="records"))

    summary: dict[
        str,
        Any,
    ] = {
        "provider": ("CPUExecutionProvider"),
        "precision": ("mixed_int8_fp32"),
        "reference_precision": ("fp32"),
        "test_rows": int(len(test_frame)),
        "model_size": {
            "fp32_bytes": (fp32_model_size_bytes),
            "int8_bytes": (int8_model_size_bytes),
            "size_reduction_percent": (size_reduction_percent),
        },
        "drift": {
            "mean_normalized_absolute": (normalized_mean_absolute_difference),
            "p95_normalized_absolute": (normalized_p95_absolute_difference),
            "p99_normalized_absolute": (normalized_p99_absolute_difference),
            "p999_normalized_absolute": (normalized_p999_absolute_difference),
            "max_normalized_absolute": (normalized_max_absolute_difference),
            "limits": {
                "mean": (max_mean_normalized_drift),
                "p99": (max_p99_normalized_drift),
                "p999": (max_p999_normalized_drift),
                "max": (max_normalized_drift),
            },
            "within_limits": {
                "mean": (mean_drift_within_limit),
                "p99": (p99_drift_within_limit),
                "p999": (p999_drift_within_limit),
                "max": (max_drift_within_limit),
            },
        },
        "predictive_quality": {
            "int8_mean_nrmse_std": (int8_mean_nrmse_std),
            "int8_mean_r2": (int8_mean_r2),
        },
        "latency": {
            "runs": (runs),
            "repeats": (repeats),
            "warmups": (warmups),
            "batch_sizes": list(usable_batch_sizes),
            "summary": (latency_summary_records),
        },
        "environment": {
            "platform": (platform()),
            "onnxruntime_version": (ort.__version__),
        },
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

    return NeuralInt8BenchmarkArtifacts(
        equivalence_path=(equivalence_path),
        task_metrics_path=(task_metrics_path),
        latency_runs_path=(latency_runs_path),
        latency_summary_path=(latency_summary_path),
        summary_path=(summary_path),
        test_rows=int(len(test_frame)),
        normalized_mean_absolute_difference=(normalized_mean_absolute_difference),
        normalized_p95_absolute_difference=(normalized_p95_absolute_difference),
        normalized_p99_absolute_difference=(normalized_p99_absolute_difference),
        normalized_p999_absolute_difference=(normalized_p999_absolute_difference),
        normalized_max_absolute_difference=(normalized_max_absolute_difference),
        int8_mean_nrmse_std=(int8_mean_nrmse_std),
        int8_mean_r2=(int8_mean_r2),
        mean_drift_within_limit=(mean_drift_within_limit),
        p99_drift_within_limit=(p99_drift_within_limit),
        p999_drift_within_limit=(p999_drift_within_limit),
        max_drift_within_limit=(max_drift_within_limit),
    )

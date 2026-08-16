"""FP16 accuracy, drift, and CoreML latency benchmarking for neural ONNX models."""

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

from edgegenbench.deployment.neural_fp16 import (
    specialize_onnx_batch_dimension,
)
from edgegenbench.evaluation.regression import (
    calculate_regression_metrics,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)

DEFAULT_FP16_BENCHMARK_BATCH_SIZES = (
    1,
    32,
    256,
)

DEFAULT_MAX_MEAN_NORMALIZED_DRIFT = 0.002
DEFAULT_MAX_NORMALIZED_DRIFT = 0.012

COREML_PROVIDER_OPTIONS = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
    "RequireStaticInputShapes": "1",
    "EnableOnSubgraphs": "0",
}


@dataclass(frozen=True)
class NeuralFp16BenchmarkArtifacts:
    """Artifacts produced by FP32-versus-FP16 CoreML benchmarking."""

    equivalence_path: Path
    task_metrics_path: Path
    latency_runs_path: Path
    latency_summary_path: Path
    summary_path: Path
    test_rows: int
    normalized_mean_absolute_difference: float
    normalized_max_absolute_difference: float
    fp16_mean_nrmse_std: float
    fp16_mean_r2: float
    mean_drift_within_limit: bool
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
    max_normalized_drift: float,
) -> None:
    """Validate benchmark iteration counts and drift guardrails."""
    if runs < 1:
        raise ValueError("runs must be at least one.")

    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    if warmups < 0:
        raise ValueError("warmups cannot be negative.")

    if max_mean_normalized_drift < 0.0:
        raise ValueError("max_mean_normalized_drift cannot be negative.")

    if max_normalized_drift < 0.0:
        raise ValueError("max_normalized_drift cannot be negative.")


def _load_test_frame(
    dataset_path: Path,
) -> pd.DataFrame:
    """Load the benchmark test split using the project split convention."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    if "split" in frame.columns:
        test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    else:
        test_frame = frame.reset_index(drop=True)

    if test_frame.empty:
        raise ValueError("No rows are available for FP16 benchmarking.")

    return test_frame


def _create_cpu_session(
    model_path: Path,
) -> ort.InferenceSession:
    """Create an ONNX Runtime CPU session."""
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


def _create_coreml_session(
    model_path: Path,
) -> ort.InferenceSession:
    """Create a static-shape CoreML session with CPU fallback."""
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    available_providers = ort.get_available_providers()

    if "CoreMLExecutionProvider" not in available_providers:
        raise RuntimeError("CoreMLExecutionProvider is unavailable.")

    providers: list[
        str
        | tuple[
            str,
            dict[str, str],
        ]
    ] = [
        (
            "CoreMLExecutionProvider",
            dict(COREML_PROVIDER_OPTIONS),
        ),
    ]

    if "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")

    return ort.InferenceSession(
        str(model_path),
        providers=providers,
    )


def _validate_session_interface(
    session: ort.InferenceSession,
    *,
    input_dim: int,
    output_dim: int,
) -> tuple[str, str]:
    """Validate the single-input/single-output neural runtime interface."""
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

    if int(input_shape[1]) != input_dim:
        raise RuntimeError("ONNX input width does not match preprocessing.")

    if int(output_shape[1]) != output_dim:
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
    """Predict a complete dynamic batch with one ONNX Runtime call."""
    (
        input_name,
        output_name,
    ) = _validate_session_interface(
        session,
        input_dim=input_dim,
        output_dim=output_dim,
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


def _predict_static_batch_one(
    session: ort.InferenceSession,
    features: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
) -> np.ndarray:
    """Predict rows with a static batch-one session."""
    (
        input_name,
        output_name,
    ) = _validate_session_interface(
        session,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    records: list[np.ndarray] = []

    for row in features:
        raw = session.run(
            [
                output_name,
            ],
            {
                input_name: row.reshape(
                    1,
                    input_dim,
                ),
            },
        )[0]

        prediction = np.asarray(
            raw,
            dtype=np.float32,
        )

        if prediction.shape != (
            1,
            output_dim,
        ):
            raise RuntimeError(f"Unexpected static batch-one output shape: {prediction.shape}.")

        records.append(prediction[0])

    stacked = np.asarray(
        records,
        dtype=np.float32,
    )

    if not np.isfinite(stacked).all():
        raise RuntimeError("CoreML prediction contains non-finite values.")

    return stacked


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
    """Measure mean and P95 latency while excluding session creation."""
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
    mean_drift: float,
    max_drift: float,
    *,
    max_mean_normalized_drift: float,
    max_normalized_drift: float,
) -> tuple[
    bool,
    bool,
]:
    """Evaluate FP16 drift against project regression guardrails."""
    return (
        bool(mean_drift <= max_mean_normalized_drift),
        bool(max_drift <= max_normalized_drift),
    )


def _summarize_latency_runs(
    latency_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate independent FP32-versus-FP16 latency runs."""
    required_columns = {
        "run",
        "batch_size",
        "fp32_mean_ms",
        "fp32_p95_ms",
        "fp16_mean_ms",
        "fp16_p95_ms",
        "fp32_over_fp16",
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

        ratios = rows["fp32_over_fp16"]

        records.append(
            {
                "batch_size": batch_size,
                "fp32_median_ms": float(rows["fp32_mean_ms"].median()),
                "fp32_median_p95_ms": float(rows["fp32_p95_ms"].median()),
                "fp16_median_ms": float(rows["fp16_mean_ms"].median()),
                "fp16_median_p95_ms": float(rows["fp16_p95_ms"].median()),
                "median_fp32_over_fp16_ratio": float(ratios.median()),
                "min_fp32_over_fp16_ratio": float(ratios.min()),
                "max_fp32_over_fp16_ratio": float(ratios.max()),
                "fp16_faster_runs": int((ratios > 1.0).sum()),
                "total_runs": int(len(rows)),
            }
        )

    return pd.DataFrame(records)


def _build_equivalence_table(
    *,
    targets: Sequence[str],
    preprocessor: NeuralPreprocessor,
    fp32_cpu_normalized: np.ndarray,
    fp32_coreml_normalized: np.ndarray,
    fp16_coreml_normalized: np.ndarray,
) -> pd.DataFrame:
    """Build target-level provider and precision drift records."""
    fp32_coreml_physical = preprocessor.inverse_transform_targets(fp32_coreml_normalized)

    fp16_coreml_physical = preprocessor.inverse_transform_targets(fp16_coreml_normalized)

    records: list[
        dict[
            str,
            float | str,
        ]
    ] = []

    for index, target in enumerate(targets):
        provider_normalized = np.abs(
            fp32_cpu_normalized[
                :,
                index,
            ]
            - fp32_coreml_normalized[
                :,
                index,
            ]
        )

        precision_normalized = np.abs(
            fp32_coreml_normalized[
                :,
                index,
            ]
            - fp16_coreml_normalized[
                :,
                index,
            ]
        )

        precision_physical = np.abs(
            fp32_coreml_physical[
                :,
                index,
            ]
            - fp16_coreml_physical[
                :,
                index,
            ]
        )

        reference_scale = float(
            np.max(
                np.abs(
                    fp32_coreml_physical[
                        :,
                        index,
                    ]
                )
            )
        )

        physical_max = float(np.max(precision_physical))

        relative_max = physical_max / reference_scale if reference_scale > 0.0 else 0.0

        records.append(
            {
                "target": str(target),
                ("fp32_cpu_vs_fp32_coreml_normalized_mean_abs"): float(
                    np.mean(provider_normalized)
                ),
                ("fp32_cpu_vs_fp32_coreml_normalized_max_abs"): float(np.max(provider_normalized)),
                ("fp32_coreml_vs_fp16_coreml_normalized_mean_abs"): float(
                    np.mean(precision_normalized)
                ),
                ("fp32_coreml_vs_fp16_coreml_normalized_max_abs"): float(
                    np.max(precision_normalized)
                ),
                ("fp32_coreml_vs_fp16_coreml_physical_mean_abs"): float(
                    np.mean(precision_physical)
                ),
                ("fp32_coreml_vs_fp16_coreml_physical_max_abs"): physical_max,
                ("fp32_coreml_vs_fp16_coreml_max_reference_relative"): float(relative_max),
            }
        )

    return pd.DataFrame(records)


def benchmark_neural_fp16(
    dataset_path: Path,
    preprocessing_path: Path,
    fp32_model_path: Path,
    fp16_model_path: Path,
    output_dir: Path = Path("artifacts/neural_fp16_benchmark"),
    batch_sizes: Sequence[int] = DEFAULT_FP16_BENCHMARK_BATCH_SIZES,
    runs: int = 5,
    repeats: int = 500,
    warmups: int = 50,
    max_mean_normalized_drift: float = (DEFAULT_MAX_MEAN_NORMALIZED_DRIFT),
    max_normalized_drift: float = (DEFAULT_MAX_NORMALIZED_DRIFT),
) -> NeuralFp16BenchmarkArtifacts:
    """Benchmark FP32 and FP16 ONNX with static-shape CoreML execution."""
    _validate_benchmark_configuration(
        runs=runs,
        repeats=repeats,
        warmups=warmups,
        max_mean_normalized_drift=(max_mean_normalized_drift),
        max_normalized_drift=(max_normalized_drift),
    )

    normalized_batch_sizes = _normalize_batch_sizes(batch_sizes)

    test_frame = _load_test_frame(dataset_path)

    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Preprocessing artifact does not exist: {preprocessing_path}")

    if not fp32_model_path.exists():
        raise FileNotFoundError(f"FP32 ONNX model does not exist: {fp32_model_path}")

    if not fp16_model_path.exists():
        raise FileNotFoundError(f"FP16 ONNX model does not exist: {fp16_model_path}")

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CoreMLExecutionProvider is unavailable.")

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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_model_dir = output_dir / "runtime_models"

    runtime_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_batch_sizes = tuple(
        sorted(
            {
                1,
                *usable_batch_sizes,
            }
        )
    )

    fp32_static_paths: dict[
        int,
        Path,
    ] = {}

    fp16_static_paths: dict[
        int,
        Path,
    ] = {}

    for batch_size in runtime_batch_sizes:
        fp32_artifact = specialize_onnx_batch_dimension(
            source_model_path=(fp32_model_path),
            output_path=(runtime_model_dir / (f"fp32_batch{batch_size}.onnx")),
            batch_size=(batch_size),
        )

        fp16_artifact = specialize_onnx_batch_dimension(
            source_model_path=(fp16_model_path),
            output_path=(runtime_model_dir / (f"fp16_batch{batch_size}.onnx")),
            batch_size=(batch_size),
        )

        fp32_static_paths[batch_size] = fp32_artifact.onnx_path

        fp16_static_paths[batch_size] = fp16_artifact.onnx_path

    fp32_cpu_session = _create_cpu_session(fp32_model_path)

    fp32_cpu_normalized = _predict_session(
        fp32_cpu_session,
        transformed_features,
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
    )

    fp32_coreml_batch1 = _create_coreml_session(fp32_static_paths[1])

    fp16_coreml_batch1 = _create_coreml_session(fp16_static_paths[1])

    fp32_coreml_normalized = _predict_static_batch_one(
        fp32_coreml_batch1,
        transformed_features,
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
    )

    fp16_coreml_normalized = _predict_static_batch_one(
        fp16_coreml_batch1,
        transformed_features,
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
    )

    provider_difference = np.abs(fp32_cpu_normalized - fp32_coreml_normalized)

    precision_difference = np.abs(fp32_coreml_normalized - fp16_coreml_normalized)

    provider_mean_drift = float(np.mean(provider_difference))

    provider_max_drift = float(np.max(provider_difference))

    normalized_mean_absolute_difference = float(np.mean(precision_difference))

    normalized_max_absolute_difference = float(np.max(precision_difference))

    (
        mean_drift_within_limit,
        max_drift_within_limit,
    ) = _evaluate_drift_limits(
        normalized_mean_absolute_difference,
        normalized_max_absolute_difference,
        max_mean_normalized_drift=(max_mean_normalized_drift),
        max_normalized_drift=(max_normalized_drift),
    )

    equivalence = _build_equivalence_table(
        targets=targets,
        preprocessor=preprocessor,
        fp32_cpu_normalized=(fp32_cpu_normalized),
        fp32_coreml_normalized=(fp32_coreml_normalized),
        fp16_coreml_normalized=(fp16_coreml_normalized),
    )

    fp16_physical = preprocessor.inverse_transform_targets(fp16_coreml_normalized)

    fp16_prediction_frame = pd.DataFrame(
        fp16_physical,
        columns=targets,
        index=test_frame.index,
    )

    task_metrics = calculate_regression_metrics(
        actual=test_frame,
        predicted=(fp16_prediction_frame),
        targets=targets,
    )

    fp16_mean_nrmse_std = float(np.nanmean(task_metrics["nrmse_std"].to_numpy(dtype=np.float64)))

    fp16_mean_r2 = float(np.nanmean(task_metrics["r2"].to_numpy(dtype=np.float64)))

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

        fp32_session = _create_coreml_session(fp32_static_paths[batch_size])

        fp16_session = _create_coreml_session(fp16_static_paths[batch_size])

        (
            fp32_input_name,
            fp32_output_name,
        ) = _validate_session_interface(
            fp32_session,
            input_dim=(preprocessor.input_dim),
            output_dim=(preprocessor.output_dim),
        )

        (
            fp16_input_name,
            fp16_output_name,
        ) = _validate_session_interface(
            fp16_session,
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

        def fp16_operation(
            session: ort.InferenceSession = (fp16_session),
            input_name: str = (fp16_input_name),
            output_name: str = (fp16_output_name),
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
                    repeats=repeats,
                    warmups=warmups,
                )

                (
                    fp16_mean_ms,
                    fp16_p95_ms,
                ) = _measure_latency(
                    fp16_operation,
                    repeats=repeats,
                    warmups=warmups,
                )
            else:
                (
                    fp16_mean_ms,
                    fp16_p95_ms,
                ) = _measure_latency(
                    fp16_operation,
                    repeats=repeats,
                    warmups=warmups,
                )

                (
                    fp32_mean_ms,
                    fp32_p95_ms,
                ) = _measure_latency(
                    fp32_operation,
                    repeats=repeats,
                    warmups=warmups,
                )

            if fp16_mean_ms <= 0.0:
                raise RuntimeError("FP16 latency must be positive.")

            latency_records.append(
                {
                    "run": run,
                    "batch_size": (batch_size),
                    "fp32_mean_ms": (fp32_mean_ms),
                    "fp32_p95_ms": (fp32_p95_ms),
                    "fp16_mean_ms": (fp16_mean_ms),
                    "fp16_p95_ms": (fp16_p95_ms),
                    "fp32_over_fp16": (fp32_mean_ms / fp16_mean_ms),
                }
            )

    latency_runs = pd.DataFrame(latency_records)

    latency_summary = _summarize_latency_runs(latency_runs)

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

    fp16_model_size_bytes = int(fp16_model_path.stat().st_size)

    if fp32_model_size_bytes < 1:
        raise RuntimeError("FP32 model is empty.")

    if fp16_model_size_bytes < 1:
        raise RuntimeError("FP16 model is empty.")

    size_reduction_percent = (1.0 - fp16_model_size_bytes / fp32_model_size_bytes) * 100.0

    latency_summary_records = json.loads(latency_summary.to_json(orient="records"))

    summary: dict[
        str,
        Any,
    ] = {
        "dataset_path": str(dataset_path),
        "preprocessing_path": str(preprocessing_path),
        "fp32_model_path": str(fp32_model_path),
        "fp16_model_path": str(fp16_model_path),
        "test_rows": int(len(test_frame)),
        "targets": list(targets),
        "precision": "fp16",
        "reference_precision": ("fp32"),
        "provider": ("CoreMLExecutionProvider"),
        "provider_options": dict(COREML_PROVIDER_OPTIONS),
        "provider_stack": (fp16_coreml_batch1.get_providers()),
        "provider_drift": {
            ("fp32_cpu_vs_fp32_coreml_mean_normalized_abs"): provider_mean_drift,
            ("fp32_cpu_vs_fp32_coreml_max_normalized_abs"): provider_max_drift,
        },
        "precision_drift": {
            ("fp32_coreml_vs_fp16_coreml_mean_normalized_abs"): (
                normalized_mean_absolute_difference
            ),
            ("fp32_coreml_vs_fp16_coreml_max_normalized_abs"): (normalized_max_absolute_difference),
            "max_mean_normalized_drift": (max_mean_normalized_drift),
            "max_normalized_drift": (max_normalized_drift),
            "mean_drift_within_limit": (mean_drift_within_limit),
            "max_drift_within_limit": (max_drift_within_limit),
        },
        "accuracy": {
            "fp16_mean_nrmse_std": (fp16_mean_nrmse_std),
            "fp16_mean_r2": (fp16_mean_r2),
        },
        "model_size": {
            "fp32_model_size_bytes": (fp32_model_size_bytes),
            "fp16_model_size_bytes": (fp16_model_size_bytes),
            "size_reduction_percent": (size_reduction_percent),
        },
        "latency": {
            "batch_sizes": list(usable_batch_sizes),
            "runs": runs,
            "repeats": repeats,
            "warmups": warmups,
            "summary": (latency_summary_records),
        },
        "environment": {
            "onnxruntime_version": (ort.__version__),
            "available_providers": (ort.get_available_providers()),
            "platform": platform(),
        },
        "artifacts": {
            "equivalence_path": str(equivalence_path),
            "task_metrics_path": str(task_metrics_path),
            "latency_runs_path": str(latency_runs_path),
            "latency_summary_path": str(latency_summary_path),
            "runtime_model_dir": str(runtime_model_dir),
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

    return NeuralFp16BenchmarkArtifacts(
        equivalence_path=(equivalence_path),
        task_metrics_path=(task_metrics_path),
        latency_runs_path=(latency_runs_path),
        latency_summary_path=(latency_summary_path),
        summary_path=(summary_path),
        test_rows=int(len(test_frame)),
        normalized_mean_absolute_difference=(normalized_mean_absolute_difference),
        normalized_max_absolute_difference=(normalized_max_absolute_difference),
        fp16_mean_nrmse_std=(fp16_mean_nrmse_std),
        fp16_mean_r2=(fp16_mean_r2),
        mean_drift_within_limit=(mean_drift_within_limit),
        max_drift_within_limit=(max_drift_within_limit),
    )

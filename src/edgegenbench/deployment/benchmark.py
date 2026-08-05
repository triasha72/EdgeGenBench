"""Equivalence and latency benchmarking for edge models."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from edgegenbench.deployment.onnx_inference import (
    OnnxFeasibilityClassifier,
    OnnxSurrogate,
)
from edgegenbench.models.feasibility import (
    FeasibilityClassifier,
)
from edgegenbench.models.preprocessing import (
    FEATURE_COLUMNS,
    validate_feature_columns,
)
from edgegenbench.models.tree_surrogate import (
    TreeSurrogate,
)


@dataclass(frozen=True)
class EdgeBenchmarkArtifacts:
    """Artifacts produced by edge benchmarking."""

    equivalence_path: Path
    latency_path: Path
    summary_path: Path
    test_rows: int
    classifier_agreement: float
    max_surrogate_absolute_error: float


def _measure_latency(
    operation: Callable[[], object],
    repeats: int,
    warmups: int,
) -> tuple[float, float]:
    """Measure mean and p95 operation latency."""
    for _ in range(warmups):
        operation()

    elapsed_times_ms: list[float] = []

    for _ in range(repeats):
        start_time = perf_counter()
        operation()
        elapsed_times_ms.append((perf_counter() - start_time) * 1000.0)

    return (
        float(np.mean(elapsed_times_ms)),
        float(
            np.percentile(
                elapsed_times_ms,
                95,
            )
        ),
    )


def benchmark_edge_models(
    dataset_path: Path,
    surrogate_model_path: Path,
    feasibility_model_path: Path,
    surrogate_onnx_path: Path,
    feasibility_onnx_path: Path,
    metadata_path: Path,
    output_dir: Path = Path("artifacts/edge_benchmark"),
    batch_sizes: Sequence[int] = (
        1,
        32,
        256,
    ),
    repeats: int = 20,
    warmups: int = 3,
) -> EdgeBenchmarkArtifacts:
    """Compare Scikit-learn and ONNX Runtime models."""
    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    if warmups < 0:
        raise ValueError("warmups cannot be negative.")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    validate_feature_columns(frame)

    if "split" in frame.columns and (frame["split"] == "test").any():
        test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    else:
        test_frame = frame.reset_index(drop=True)

    if test_frame.empty:
        raise ValueError("No rows are available for edge benchmarking.")

    sklearn_surrogate = TreeSurrogate.load(surrogate_model_path)

    sklearn_classifier = FeasibilityClassifier.load(feasibility_model_path)

    onnx_surrogate = OnnxSurrogate.load(
        surrogate_onnx_path,
        metadata_path,
    )

    onnx_classifier = OnnxFeasibilityClassifier.load(
        feasibility_onnx_path,
        metadata_path,
    )

    sklearn_surrogate_predictions = sklearn_surrogate.predict(test_frame)

    onnx_surrogate_predictions = onnx_surrogate.predict(test_frame)

    equivalence_records: list[dict[str, float | str]] = []

    for target in sklearn_surrogate.targets:
        absolute_errors = np.abs(
            sklearn_surrogate_predictions[target].to_numpy(dtype=np.float64)
            - onnx_surrogate_predictions[target].to_numpy(dtype=np.float64)
        )

        equivalence_records.append(
            {
                "model": "surrogate",
                "output": target,
                "mean_absolute_difference": float(np.mean(absolute_errors)),
                "max_absolute_difference": float(np.max(absolute_errors)),
            }
        )

    sklearn_probabilities = sklearn_classifier.predict_feasibility_probability(test_frame).to_numpy(
        dtype=np.float64
    )

    onnx_probabilities = onnx_classifier.predict_feasibility_probability(test_frame).to_numpy(
        dtype=np.float64
    )

    probability_differences = np.abs(sklearn_probabilities - onnx_probabilities)

    equivalence_records.append(
        {
            "model": "feasibility",
            "output": ("feasibility_probability"),
            "mean_absolute_difference": float(np.mean(probability_differences)),
            "max_absolute_difference": float(np.max(probability_differences)),
        }
    )

    sklearn_classes = sklearn_probabilities >= sklearn_classifier.threshold

    onnx_classes = onnx_probabilities >= onnx_classifier.threshold

    classifier_agreement = float(np.mean(sklearn_classes == onnx_classes))

    equivalence = pd.DataFrame(equivalence_records)

    latency_records: list[dict[str, float | int | str]] = []

    normalized_batch_sizes = tuple(
        sorted({int(batch_size) for batch_size in batch_sizes if int(batch_size) > 0})
    )

    operations = (
        (
            "scikit_learn",
            "surrogate",
            sklearn_surrogate.predict,
        ),
        (
            "onnx_runtime",
            "surrogate",
            onnx_surrogate.predict,
        ),
        (
            "scikit_learn",
            "feasibility",
            (sklearn_classifier.predict_feasibility_probability),
        ),
        (
            "onnx_runtime",
            "feasibility",
            (onnx_classifier.predict_feasibility_probability),
        ),
    )

    for batch_size in normalized_batch_sizes:
        if batch_size > len(test_frame):
            continue

        batch = (
            test_frame.iloc[:batch_size]
            .loc[
                :,
                list(FEATURE_COLUMNS),
            ]
            .copy()
        )

        for runtime_name, model_name, operation in operations:
            mean_latency_ms, p95_latency_ms = _measure_latency(
                operation=(lambda operation=operation, batch=batch: operation(batch)),
                repeats=repeats,
                warmups=warmups,
            )

            latency_records.append(
                {
                    "runtime": runtime_name,
                    "model": model_name,
                    "batch_size": (batch_size),
                    "repeats": repeats,
                    "mean_batch_latency_ms": (mean_latency_ms),
                    "p95_batch_latency_ms": (p95_latency_ms),
                    "mean_sample_latency_us": (mean_latency_ms * 1000.0 / batch_size),
                }
            )

    latency = pd.DataFrame(latency_records)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    equivalence_path = output_dir / "equivalence.csv"
    latency_path = output_dir / "latency.csv"
    summary_path = output_dir / "summary.json"

    equivalence.to_csv(
        equivalence_path,
        index=False,
    )

    latency.to_csv(
        latency_path,
        index=False,
    )

    max_surrogate_absolute_error = float(
        equivalence.loc[
            equivalence["model"] == "surrogate",
            "max_absolute_difference",
        ].max()
    )

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "test_rows": int(len(test_frame)),
        "classifier_agreement": (classifier_agreement),
        "max_surrogate_absolute_error": (max_surrogate_absolute_error),
        "max_probability_absolute_error": float(np.max(probability_differences)),
        "source_model_sizes_bytes": {
            "surrogate": int(surrogate_model_path.stat().st_size),
            "feasibility": int(feasibility_model_path.stat().st_size),
        },
        "onnx_model_sizes_bytes": {
            "surrogate": int(surrogate_onnx_path.stat().st_size),
            "feasibility": int(feasibility_onnx_path.stat().st_size),
        },
        "equivalence_path": str(equivalence_path),
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

    return EdgeBenchmarkArtifacts(
        equivalence_path=equivalence_path,
        latency_path=latency_path,
        summary_path=summary_path,
        test_rows=len(test_frame),
        classifier_agreement=(classifier_agreement),
        max_surrogate_absolute_error=(max_surrogate_absolute_error),
    )

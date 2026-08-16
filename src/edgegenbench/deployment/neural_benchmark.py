"""Equivalence and latency benchmarking for the neural ONNX surrogate."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from edgegenbench.deployment.neural_onnx_inference import (
    NeuralOnnxSurrogate,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)
from edgegenbench.models.neural_surrogate import (
    load_neural_surrogate_checkpoint,
)


@dataclass(frozen=True)
class NeuralBenchmarkArtifacts:
    """Artifacts produced by neural deployment benchmarking."""

    equivalence_path: Path
    latency_path: Path
    summary_path: Path
    test_rows: int
    normalized_mean_absolute_difference: float
    normalized_max_absolute_difference: float
    normalized_equivalent: bool


def _measure_latency(
    operation: Callable[[], object],
    repeats: int,
    warmups: int,
) -> tuple[float, float]:
    """Measure mean and P95 operation latency."""
    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    if warmups < 0:
        raise ValueError("warmups cannot be negative.")

    for _ in range(warmups):
        operation()

    elapsed_ms: list[float] = []

    for _ in range(repeats):
        start = perf_counter()

        operation()

        elapsed_ms.append((perf_counter() - start) * 1000.0)

    return (
        float(np.mean(elapsed_ms)),
        float(
            np.percentile(
                elapsed_ms,
                95,
            )
        ),
    )


def benchmark_neural_onnx(
    dataset_path: Path,
    model_path: Path,
    preprocessing_path: Path,
    onnx_model_path: Path,
    metadata_path: Path,
    output_dir: Path = Path("artifacts/neural_onnx_benchmark"),
    batch_sizes: Sequence[int] = (
        1,
        32,
        256,
    ),
    repeats: int = 200,
    warmups: int = 20,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-5,
) -> NeuralBenchmarkArtifacts:
    """Benchmark PyTorch CPU against ONNX Runtime CPU."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    if warmups < 0:
        raise ValueError("warmups cannot be negative.")

    frame = pd.read_csv(dataset_path)

    if "split" in frame.columns:
        test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    else:
        test_frame = frame.reset_index(drop=True)

    if test_frame.empty:
        raise ValueError("No rows are available for neural deployment benchmarking.")

    preprocessor = NeuralPreprocessor.load(preprocessing_path)

    pytorch_model, targets = load_neural_surrogate_checkpoint(model_path)

    pytorch_model = pytorch_model.cpu()

    pytorch_model.eval()

    runtime = NeuralOnnxSurrogate.load(
        model_path=onnx_model_path,
        metadata_path=metadata_path,
        preprocessing_path=(preprocessing_path),
    )

    transformed_features = preprocessor.transform_features(test_frame)

    pytorch_features = torch.from_numpy(transformed_features).to(
        dtype=torch.float32,
        device="cpu",
    )

    with torch.inference_mode():
        pytorch_normalized = pytorch_model(pytorch_features).cpu().numpy().astype(np.float32)

    onnx_normalized = runtime.session.run(
        [
            runtime.output_name,
        ],
        {
            runtime.input_name: (transformed_features),
        },
    )[0]

    onnx_normalized = np.asarray(
        onnx_normalized,
        dtype=np.float32,
    )

    normalized_difference = np.abs(pytorch_normalized - onnx_normalized)

    normalized_mean_absolute_difference = float(np.mean(normalized_difference))

    normalized_max_absolute_difference = float(np.max(normalized_difference))

    normalized_equivalent = bool(
        np.allclose(
            pytorch_normalized,
            onnx_normalized,
            rtol=rtol,
            atol=atol,
        )
    )

    pytorch_physical = preprocessor.inverse_transform_targets(pytorch_normalized)

    onnx_physical = preprocessor.inverse_transform_targets(onnx_normalized)

    equivalence_records: list[dict[str, float | str]] = []

    for index, target in enumerate(targets):
        differences = np.abs(
            pytorch_physical[
                :,
                index,
            ]
            - onnx_physical[
                :,
                index,
            ]
        )

        reference_scale = float(
            np.max(
                np.abs(
                    pytorch_physical[
                        :,
                        index,
                    ]
                )
            )
        )

        maximum_difference = float(np.max(differences))

        relative_maximum_difference = (
            maximum_difference / reference_scale if reference_scale > 0.0 else 0.0
        )

        equivalence_records.append(
            {
                "target": target,
                "mean_absolute_difference": float(np.mean(differences)),
                "max_absolute_difference": (maximum_difference),
                "max_reference_relative_difference": (relative_maximum_difference),
            }
        )

    equivalence = pd.DataFrame(equivalence_records)

    normalized_batch_sizes = tuple(
        sorted({int(batch_size) for batch_size in batch_sizes if int(batch_size) > 0})
    )

    latency_records: list[dict[str, float | int | str]] = []

    for batch_size in normalized_batch_sizes:
        if batch_size > len(transformed_features):
            continue

        numpy_batch = transformed_features[:batch_size].astype(
            np.float32,
            copy=False,
        )

        torch_batch = torch.from_numpy(numpy_batch)

        def pytorch_operation(
            batch: torch.Tensor = torch_batch,
        ) -> object:
            return pytorch_model(batch)

        def onnx_operation(
            batch: np.ndarray = numpy_batch,
        ) -> object:
            return runtime.session.run(
                [
                    runtime.output_name,
                ],
                {
                    runtime.input_name: batch,
                },
            )

        with torch.inference_mode():
            (
                pytorch_mean_ms,
                pytorch_p95_ms,
            ) = _measure_latency(
                pytorch_operation,
                repeats=repeats,
                warmups=warmups,
            )

        (
            onnx_mean_ms,
            onnx_p95_ms,
        ) = _measure_latency(
            onnx_operation,
            repeats=repeats,
            warmups=warmups,
        )

        latency_records.extend(
            [
                {
                    "runtime": "pytorch_cpu",
                    "batch_size": batch_size,
                    "repeats": repeats,
                    "warmups": warmups,
                    "mean_batch_latency_ms": (pytorch_mean_ms),
                    "p95_batch_latency_ms": (pytorch_p95_ms),
                    "mean_sample_latency_us": (pytorch_mean_ms * 1000.0 / batch_size),
                },
                {
                    "runtime": ("onnxruntime_cpu"),
                    "batch_size": batch_size,
                    "repeats": repeats,
                    "warmups": warmups,
                    "mean_batch_latency_ms": (onnx_mean_ms),
                    "p95_batch_latency_ms": (onnx_p95_ms),
                    "mean_sample_latency_us": (onnx_mean_ms * 1000.0 / batch_size),
                },
            ]
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

    speedup_by_batch: dict[
        str,
        float,
    ] = {}

    for batch_size in normalized_batch_sizes:
        batch_rows = latency.loc[latency["batch_size"] == batch_size]

        if len(batch_rows) != 2:
            continue

        pytorch_rows = batch_rows.loc[batch_rows["runtime"] == "pytorch_cpu"]

        onnx_rows = batch_rows.loc[batch_rows["runtime"] == "onnxruntime_cpu"]

        if len(pytorch_rows) == 1 and len(onnx_rows) == 1:
            pytorch_latency = float(pytorch_rows["mean_batch_latency_ms"].iloc[0])

            onnx_latency = float(onnx_rows["mean_batch_latency_ms"].iloc[0])

            if onnx_latency > 0.0:
                speedup_by_batch[str(batch_size)] = pytorch_latency / onnx_latency

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "preprocessing_path": str(preprocessing_path),
        "onnx_model_path": str(onnx_model_path),
        "metadata_path": str(metadata_path),
        "test_rows": int(len(test_frame)),
        "targets": list(targets),
        "normalized_mean_absolute_difference": (normalized_mean_absolute_difference),
        "normalized_max_absolute_difference": (normalized_max_absolute_difference),
        "normalized_equivalent": (normalized_equivalent),
        "rtol": rtol,
        "atol": atol,
        "batch_sizes": list(normalized_batch_sizes),
        "repeats": repeats,
        "warmups": warmups,
        "pytorch_model_size_bytes": int(model_path.stat().st_size),
        "onnx_model_size_bytes": int(onnx_model_path.stat().st_size),
        "speedup_pytorch_over_onnx_by_batch": (speedup_by_batch),
        "torch_version": (torch.__version__),
        "torch_cpu_threads": (torch.get_num_threads()),
        "onnxruntime_providers": (runtime.session.get_providers()),
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

    return NeuralBenchmarkArtifacts(
        equivalence_path=(equivalence_path),
        latency_path=latency_path,
        summary_path=summary_path,
        test_rows=len(test_frame),
        normalized_mean_absolute_difference=(normalized_mean_absolute_difference),
        normalized_max_absolute_difference=(normalized_max_absolute_difference),
        normalized_equivalent=(normalized_equivalent),
    )

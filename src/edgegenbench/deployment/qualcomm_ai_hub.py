"""Pure helpers for Qualcomm AI Hub and QNN deployment evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuntimeParityMetrics:
    """Numerical differences between reference and deployed predictions."""

    sample_count: int
    output_width: int
    mae: float
    rmse: float
    max_abs_error: float
    mean_normalized_drift: float
    max_normalized_drift: float
    allclose_rtol_1e3_atol_1e3: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class QnnProfileSummary:
    """Normalized Qualcomm QNN profile information."""

    batch_size: int
    estimated_inference_time_us: float | None
    estimated_inference_latency_ms: float | None
    estimated_throughput_samples_per_second: float | None
    estimated_inference_peak_memory_bytes: int | None
    compute_units: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class QualcommInt8Acceptance:
    """Evidence gate for a measured Qualcomm-native INT8 candidate."""

    source_model_sha256: str
    quantized_model_sha256: str
    calibration_partition: str
    calibration_sample_count: int
    profiled_batch_sizes: tuple[int, ...]
    parity: RuntimeParityMetrics
    max_normalized_drift_limit: float
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def assess_qualcomm_int8_candidate(
    *,
    source_model_sha256: str,
    quantized_model_sha256: str,
    calibration_partition: str,
    calibration_sample_count: int,
    profiles: tuple[QnnProfileSummary, ...],
    parity: RuntimeParityMetrics,
    required_batch_sizes: tuple[int, ...] = (1, 32, 256),
    max_normalized_drift: float = 0.01,
) -> QualcommInt8Acceptance:
    """Accept an INT8 claim only when provenance, NPU profiles, and parity pass."""

    reasons: list[str] = []
    hexadecimal = set("0123456789abcdef")
    hashes = (source_model_sha256.lower(), quantized_model_sha256.lower())
    if any(len(value) != 64 or set(value) - hexadecimal for value in hashes):
        reasons.append("model hashes must be lowercase-compatible SHA-256 values")
    if hashes[0] == hashes[1]:
        reasons.append("source and quantized model hashes must be distinct")
    if calibration_partition.strip().lower() != "train":
        reasons.append("quantization calibration must use the training partition only")
    if calibration_sample_count <= 0:
        reasons.append("calibration_sample_count must be positive")
    if max_normalized_drift <= 0.0:
        raise ValueError("max_normalized_drift must be positive.")

    batches = tuple(sorted(profile.batch_size for profile in profiles))
    if batches != tuple(sorted(required_batch_sizes)):
        reasons.append("all required batch profiles must be present exactly once")
    for profile in profiles:
        if profile.estimated_inference_time_us is None:
            reasons.append(f"batch {profile.batch_size} is missing measured latency")
        if profile.estimated_inference_peak_memory_bytes is None:
            reasons.append(f"batch {profile.batch_size} is missing measured peak memory")
        non_npu = {
            unit: count for unit, count in profile.compute_units.items() if unit != "NPU" and count
        }
        if profile.compute_units.get("NPU", 0) <= 0 or non_npu:
            reasons.append(f"batch {profile.batch_size} does not show exclusive NPU placement")
    if parity.max_normalized_drift > max_normalized_drift:
        reasons.append("held-out normalized drift exceeds the preregistered limit")

    return QualcommInt8Acceptance(
        source_model_sha256=hashes[0],
        quantized_model_sha256=hashes[1],
        calibration_partition=calibration_partition.strip().lower(),
        calibration_sample_count=calibration_sample_count,
        profiled_batch_sizes=batches,
        parity=parity,
        max_normalized_drift_limit=max_normalized_drift,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def qnn_graph_option(graph_name: str) -> str:
    """Return the AI Hub option used to select one linked QNN graph."""
    normalized = graph_name.strip()

    if not normalized:
        raise ValueError("graph_name cannot be empty.")

    forbidden = {" ", "\t", "\n", "\r", ";"}

    if any(character in normalized for character in forbidden):
        raise ValueError(
            "graph_name contains characters that are not supported by the "
            "EdgeGenBench QNN graph-selection helper."
        )

    return f"--qnn_options context_enable_graphs={normalized}"


def stringify_metadata(
    metadata: Mapping[object, object],
) -> dict[str, str]:
    """Convert AI Hub enum-keyed metadata into stable strings."""
    return {str(key): str(value) for key, value in metadata.items()}


def collect_compute_units(
    payload: object,
) -> dict[str, int]:
    """Count compute-unit labels recursively in an AI Hub profile payload."""
    counts: Counter[str] = Counter()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            compute_unit = value.get("compute_unit")

            if compute_unit is not None:
                counts[str(compute_unit)] += 1

            for child in value.values():
                visit(child)

        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)

    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _optional_float(
    value: Any,
) -> float | None:
    """Convert an external numeric value to float."""
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError("Expected a numeric profile value, received bool.")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a float-compatible profile value, received {value!r}.") from exc


def _optional_int(
    value: Any,
) -> int | None:
    """Convert an external numeric value to int."""
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError("Expected an integer profile value, received bool.")

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an int-compatible profile value, received {value!r}.") from exc


def summarize_profile(
    profile: Mapping[str, Any],
    batch_size: int,
) -> QnnProfileSummary:
    """Normalize the performance fields used by EdgeGenBench."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    execution_summary_value = profile.get("execution_summary", profile)

    if not isinstance(execution_summary_value, Mapping):
        raise ValueError("AI Hub profile execution_summary must be a mapping.")

    estimated_us = _optional_float(execution_summary_value.get("estimated_inference_time"))
    peak_memory = _optional_int(execution_summary_value.get("estimated_inference_peak_memory"))

    latency_ms: float | None = None
    throughput: float | None = None

    if estimated_us is not None:
        if estimated_us <= 0.0:
            raise ValueError("estimated_inference_time must be positive when present.")

        latency_ms = estimated_us / 1000.0
        throughput = float(batch_size) * 1_000_000.0 / estimated_us

    return QnnProfileSummary(
        batch_size=batch_size,
        estimated_inference_time_us=estimated_us,
        estimated_inference_latency_ms=latency_ms,
        estimated_throughput_samples_per_second=throughput,
        estimated_inference_peak_memory_bytes=peak_memory,
        compute_units=collect_compute_units(profile),
    )


def calculate_runtime_parity(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> RuntimeParityMetrics:
    """Calculate deployment drift against a reference output tensor."""
    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)

    if reference_array.ndim != 2:
        raise ValueError("reference predictions must have two dimensions.")

    if candidate_array.ndim != 2:
        raise ValueError("candidate predictions must have two dimensions.")

    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Prediction-shape mismatch: "
            f"reference={reference_array.shape}, "
            f"candidate={candidate_array.shape}."
        )

    if reference_array.shape[0] < 1:
        raise ValueError("At least one prediction row is required.")

    difference = candidate_array - reference_array
    absolute_error = np.abs(difference)

    mae = float(np.mean(absolute_error, dtype=np.float64))
    rmse = float(
        np.sqrt(
            np.mean(
                np.square(difference, dtype=np.float64),
                dtype=np.float64,
            )
        )
    )
    max_abs_error = float(np.max(absolute_error))

    output_scale = np.std(reference_array, axis=0, dtype=np.float64)
    output_scale = np.maximum(output_scale, np.float64(1.0e-8))
    normalized_drift = absolute_error / output_scale[None, :]

    mean_normalized_drift = float(np.mean(normalized_drift, dtype=np.float64))
    max_normalized_drift = float(np.max(normalized_drift))

    allclose = bool(
        np.allclose(
            reference_array,
            candidate_array,
            rtol=1.0e-3,
            atol=1.0e-3,
        )
    )

    return RuntimeParityMetrics(
        sample_count=int(reference_array.shape[0]),
        output_width=int(reference_array.shape[1]),
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs_error,
        mean_normalized_drift=mean_normalized_drift,
        max_normalized_drift=max_normalized_drift,
        allclose_rtol_1e3_atol_1e3=allclose,
    )

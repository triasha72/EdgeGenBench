"""Constraint-aware selection for validated neural deployment candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

SelectionPolicy = Literal[
    "lowest_latency",
    "smallest_model",
    "highest_accuracy",
    "balanced",
]

_VALID_POLICIES: tuple[SelectionPolicy, ...] = (
    "lowest_latency",
    "smallest_model",
    "highest_accuracy",
    "balanced",
)


@dataclass(frozen=True)
class DeploymentCandidate:
    """One measured model/provider/batch deployment candidate."""

    name: str
    precision: str
    provider: str
    model_path: Path
    model_size_bytes: int
    batch_size: int
    median_latency_ms: float
    mean_r2: float
    mean_nrmse_std: float
    max_normalized_drift: float
    benchmark_source: Path
    measurement_context: str

    def __post_init__(self) -> None:
        """Validate deployment-candidate invariants."""
        for field_name, value in (
            ("name", self.name),
            ("precision", self.precision),
            ("provider", self.provider),
            ("measurement_context", self.measurement_context),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty.")

        if self.model_size_bytes < 1:
            raise ValueError("model_size_bytes must be positive.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

        _require_finite_nonnegative(
            self.median_latency_ms,
            "median_latency_ms",
            allow_zero=False,
        )
        _require_finite(self.mean_r2, "mean_r2")
        _require_finite_nonnegative(
            self.mean_nrmse_std,
            "mean_nrmse_std",
        )
        _require_finite_nonnegative(
            self.max_normalized_drift,
            "max_normalized_drift",
        )


@dataclass(frozen=True)
class DeploymentConstraints:
    """Hard deployment constraints applied before ranking candidates."""

    batch_size: int
    max_latency_ms: float | None = None
    max_model_size_mb: float | None = None
    min_r2: float | None = None
    max_nrmse_std: float | None = None
    max_normalized_drift: float | None = None
    required_provider: str | None = None

    def __post_init__(self) -> None:
        """Validate constraint values."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

        for field_name, value in (
            ("max_latency_ms", self.max_latency_ms),
            ("max_model_size_mb", self.max_model_size_mb),
            ("max_nrmse_std", self.max_nrmse_std),
            ("max_normalized_drift", self.max_normalized_drift),
        ):
            if value is not None:
                _require_finite_nonnegative(value, field_name)

        if self.min_r2 is not None:
            _require_finite(self.min_r2, "min_r2")

        if self.required_provider is not None and not self.required_provider.strip():
            raise ValueError("required_provider cannot be blank.")


@dataclass(frozen=True)
class CandidateDecision:
    """Feasibility and ranking information for one candidate."""

    candidate: DeploymentCandidate
    feasible: bool
    rejection_reasons: tuple[str, ...]
    score: float | None = None


@dataclass(frozen=True)
class SelectionResult:
    """Complete deployment-selection result."""

    policy: SelectionPolicy
    constraints: DeploymentConstraints
    selected_candidate: DeploymentCandidate | None
    decisions: tuple[CandidateDecision, ...]

    @property
    def feasible_count(self) -> int:
        """Return the number of candidates that satisfy every hard constraint."""
        return sum(decision.feasible for decision in self.decisions)


@dataclass(frozen=True)
class DeploymentSelectionArtifacts:
    """Files written for one deployment-selection decision."""

    json_path: Path
    markdown_path: Path


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def _require_finite_nonnegative(
    value: float,
    name: str,
    *,
    allow_zero: bool = True,
) -> None:
    _require_finite(value, name)
    if allow_zero:
        if value < 0.0:
            raise ValueError(f"{name} cannot be negative.")
    elif value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object with a useful error when the artifact is invalid."""
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact does not exist: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return cast(dict[str, Any], raw)


def _mapping(parent: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object '{key}' in {source}.")
    return cast(dict[str, Any], value)


def _records(parent: dict[str, Any], key: str, source: Path) -> list[dict[str, Any]]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list '{key}' in {source}.")

    records: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at {key}[{index}] in {source}.")
        records.append(cast(dict[str, Any], record))
    return records


def _number(parent: dict[str, Any], key: str, source: Path) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric '{key}' in {source}.")
    result = float(value)
    _require_finite(result, key)
    return result


def _positive_int(parent: dict[str, Any], key: str, source: Path) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer '{key}' in {source}.")
    if value < 1:
        raise ValueError(f"Expected positive '{key}' in {source}.")
    return value


def _string(parent: dict[str, Any], key: str, source: Path) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string '{key}' in {source}.")
    return value


def _validate_model_size(
    model_path: Path,
    *,
    expected_bytes: int,
    label: str,
) -> int:
    if not model_path.exists():
        raise FileNotFoundError(f"{label} model does not exist: {model_path}")

    actual_bytes = int(model_path.stat().st_size)
    if actual_bytes < 1:
        raise ValueError(f"{label} model is empty: {model_path}")

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{label} model-size mismatch: summary={expected_bytes} bytes, "
            f"file={actual_bytes} bytes ({model_path})."
        )

    return actual_bytes


def _latency_rows(
    summary: dict[str, Any],
    *,
    source: Path,
) -> list[dict[str, Any]]:
    latency = _mapping(summary, "latency", source)
    return _records(latency, "summary", source)


def load_edgegenbench_deployment_candidates(
    neural_training_summary_path: Path = Path("artifacts/neural_surrogate/summary.json"),
    fp16_benchmark_summary_path: Path = Path("artifacts/neural_fp16_benchmark/summary.json"),
    int8_benchmark_summary_path: Path = Path("artifacts/neural_int8_benchmark/summary.json"),
    fp32_model_path: Path = Path("artifacts/neural_onnx/neural_surrogate.onnx"),
    fp16_model_path: Path = Path("artifacts/neural_fp16/neural_surrogate_fp16.onnx"),
    int8_model_path: Path = Path("artifacts/neural_int8/neural_surrogate_int8.onnx"),
) -> tuple[DeploymentCandidate, ...]:
    """Build provider-aware candidates from existing validated benchmark artifacts."""
    training_summary = _load_json_object(neural_training_summary_path)
    fp16_summary = _load_json_object(fp16_benchmark_summary_path)
    int8_summary = _load_json_object(int8_benchmark_summary_path)

    fp32_mean_r2 = _number(training_summary, "mean_test_r2", neural_training_summary_path)
    fp32_mean_nrmse = _number(
        training_summary,
        "mean_test_nrmse_std",
        neural_training_summary_path,
    )

    fp16_accuracy = _mapping(fp16_summary, "accuracy", fp16_benchmark_summary_path)
    fp16_mean_r2 = _number(fp16_accuracy, "fp16_mean_r2", fp16_benchmark_summary_path)
    fp16_mean_nrmse = _number(
        fp16_accuracy,
        "fp16_mean_nrmse_std",
        fp16_benchmark_summary_path,
    )

    int8_quality = _mapping(int8_summary, "predictive_quality", int8_benchmark_summary_path)
    int8_mean_r2 = _number(int8_quality, "int8_mean_r2", int8_benchmark_summary_path)
    int8_mean_nrmse = _number(
        int8_quality,
        "int8_mean_nrmse_std",
        int8_benchmark_summary_path,
    )

    fp16_size = _mapping(fp16_summary, "model_size", fp16_benchmark_summary_path)
    int8_size = _mapping(int8_summary, "model_size", int8_benchmark_summary_path)

    fp32_bytes_fp16 = _positive_int(
        fp16_size,
        "fp32_model_size_bytes",
        fp16_benchmark_summary_path,
    )
    fp32_bytes_int8 = _positive_int(
        int8_size,
        "fp32_bytes",
        int8_benchmark_summary_path,
    )

    if fp32_bytes_fp16 != fp32_bytes_int8:
        raise ValueError(
            "FP32 model-size metadata disagrees between FP16 and INT8 benchmark summaries."
        )

    fp32_bytes = _validate_model_size(
        fp32_model_path,
        expected_bytes=fp32_bytes_int8,
        label="FP32",
    )
    fp16_bytes = _validate_model_size(
        fp16_model_path,
        expected_bytes=_positive_int(
            fp16_size,
            "fp16_model_size_bytes",
            fp16_benchmark_summary_path,
        ),
        label="FP16",
    )
    int8_bytes = _validate_model_size(
        int8_model_path,
        expected_bytes=_positive_int(
            int8_size,
            "int8_bytes",
            int8_benchmark_summary_path,
        ),
        label="INT8",
    )

    fp16_provider = _string(fp16_summary, "provider", fp16_benchmark_summary_path)
    int8_provider = _string(int8_summary, "provider", int8_benchmark_summary_path)

    fp16_provider_drift = _mapping(
        fp16_summary,
        "provider_drift",
        fp16_benchmark_summary_path,
    )
    fp16_precision_drift = _mapping(
        fp16_summary,
        "precision_drift",
        fp16_benchmark_summary_path,
    )
    int8_drift = _mapping(int8_summary, "drift", int8_benchmark_summary_path)

    fp32_coreml_max_drift = _number(
        fp16_provider_drift,
        "fp32_cpu_vs_fp32_coreml_max_normalized_abs",
        fp16_benchmark_summary_path,
    )
    fp16_max_drift = _number(
        fp16_precision_drift,
        "fp32_coreml_vs_fp16_coreml_max_normalized_abs",
        fp16_benchmark_summary_path,
    )
    int8_max_drift = _number(
        int8_drift,
        "max_normalized_absolute",
        int8_benchmark_summary_path,
    )

    candidates: list[DeploymentCandidate] = []

    for row in _latency_rows(int8_summary, source=int8_benchmark_summary_path):
        batch_size = _positive_int(row, "batch_size", int8_benchmark_summary_path)

        candidates.extend(
            [
                DeploymentCandidate(
                    name="fp32_cpu",
                    precision="fp32",
                    provider=int8_provider,
                    model_path=fp32_model_path,
                    model_size_bytes=fp32_bytes,
                    batch_size=batch_size,
                    median_latency_ms=_number(
                        row,
                        "fp32_median_ms",
                        int8_benchmark_summary_path,
                    ),
                    mean_r2=fp32_mean_r2,
                    mean_nrmse_std=fp32_mean_nrmse,
                    max_normalized_drift=0.0,
                    benchmark_source=int8_benchmark_summary_path,
                    measurement_context="paired FP32/INT8 ONNX Runtime CPU benchmark",
                ),
                DeploymentCandidate(
                    name="mixed_int8_fp32_cpu",
                    precision="mixed_int8_fp32",
                    provider=int8_provider,
                    model_path=int8_model_path,
                    model_size_bytes=int8_bytes,
                    batch_size=batch_size,
                    median_latency_ms=_number(
                        row,
                        "int8_median_ms",
                        int8_benchmark_summary_path,
                    ),
                    mean_r2=int8_mean_r2,
                    mean_nrmse_std=int8_mean_nrmse,
                    max_normalized_drift=int8_max_drift,
                    benchmark_source=int8_benchmark_summary_path,
                    measurement_context="paired FP32/INT8 ONNX Runtime CPU benchmark",
                ),
            ]
        )

    for row in _latency_rows(fp16_summary, source=fp16_benchmark_summary_path):
        batch_size = _positive_int(row, "batch_size", fp16_benchmark_summary_path)

        candidates.extend(
            [
                DeploymentCandidate(
                    name="fp32_coreml",
                    precision="fp32",
                    provider=fp16_provider,
                    model_path=fp32_model_path,
                    model_size_bytes=fp32_bytes,
                    batch_size=batch_size,
                    median_latency_ms=_number(
                        row,
                        "fp32_median_ms",
                        fp16_benchmark_summary_path,
                    ),
                    mean_r2=fp32_mean_r2,
                    mean_nrmse_std=fp32_mean_nrmse,
                    max_normalized_drift=fp32_coreml_max_drift,
                    benchmark_source=fp16_benchmark_summary_path,
                    measurement_context="paired FP32/FP16 CoreML benchmark",
                ),
                DeploymentCandidate(
                    name="fp16_coreml",
                    precision="fp16",
                    provider=fp16_provider,
                    model_path=fp16_model_path,
                    model_size_bytes=fp16_bytes,
                    batch_size=batch_size,
                    median_latency_ms=_number(
                        row,
                        "fp16_median_ms",
                        fp16_benchmark_summary_path,
                    ),
                    mean_r2=fp16_mean_r2,
                    mean_nrmse_std=fp16_mean_nrmse,
                    max_normalized_drift=fp16_max_drift,
                    benchmark_source=fp16_benchmark_summary_path,
                    measurement_context="paired FP32/FP16 CoreML benchmark",
                ),
            ]
        )

    if not candidates:
        raise ValueError("No deployment candidates were found in the benchmark summaries.")

    key_count = len({(candidate.name, candidate.batch_size) for candidate in candidates})
    if key_count != len(candidates):
        raise ValueError(
            "Deployment benchmark summaries produced duplicate candidate/batch records."
        )

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.batch_size,
                candidate.provider,
                candidate.name,
            ),
        )
    )


def evaluate_candidate(
    candidate: DeploymentCandidate,
    constraints: DeploymentConstraints,
) -> CandidateDecision:
    """Apply hard deployment constraints to one candidate."""
    reasons: list[str] = []

    if candidate.batch_size != constraints.batch_size:
        reasons.append(
            f"batch size {candidate.batch_size} does not match required {constraints.batch_size}"
        )

    if (
        constraints.required_provider is not None
        and candidate.provider != constraints.required_provider
    ):
        reasons.append(
            f"provider {candidate.provider!r} does not match required "
            f"{constraints.required_provider!r}"
        )

    if (
        constraints.max_latency_ms is not None
        and candidate.median_latency_ms > constraints.max_latency_ms
    ):
        reasons.append(
            f"median latency {candidate.median_latency_ms:.6f} ms exceeds "
            f"{constraints.max_latency_ms:.6f} ms"
        )

    if constraints.max_model_size_mb is not None:
        model_size_mb = candidate.model_size_bytes / (1024.0 * 1024.0)
        if model_size_mb > constraints.max_model_size_mb:
            reasons.append(
                f"model size {model_size_mb:.6f} MB exceeds {constraints.max_model_size_mb:.6f} MB"
            )

    if constraints.min_r2 is not None and candidate.mean_r2 < constraints.min_r2:
        reasons.append(f"mean R2 {candidate.mean_r2:.6f} is below {constraints.min_r2:.6f}")

    if (
        constraints.max_nrmse_std is not None
        and candidate.mean_nrmse_std > constraints.max_nrmse_std
    ):
        reasons.append(
            f"mean NRMSE {candidate.mean_nrmse_std:.6f} exceeds {constraints.max_nrmse_std:.6f}"
        )

    if (
        constraints.max_normalized_drift is not None
        and candidate.max_normalized_drift > constraints.max_normalized_drift
    ):
        reasons.append(
            f"maximum normalized drift {candidate.max_normalized_drift:.6f} exceeds "
            f"{constraints.max_normalized_drift:.6f}"
        )

    return CandidateDecision(
        candidate=candidate,
        feasible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _min_max_normalize(value: float, values: list[float]) -> float:
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        return 0.0
    return (value - low) / (high - low)


def _balanced_scores(
    candidates: list[DeploymentCandidate],
) -> dict[tuple[str, int], float]:
    latencies = [candidate.median_latency_ms for candidate in candidates]
    sizes = [float(candidate.model_size_bytes) for candidate in candidates]
    nrmses = [candidate.mean_nrmse_std for candidate in candidates]
    drifts = [candidate.max_normalized_drift for candidate in candidates]

    scores: dict[tuple[str, int], float] = {}
    for candidate in candidates:
        score = (
            0.40 * _min_max_normalize(candidate.median_latency_ms, latencies)
            + 0.25 * _min_max_normalize(float(candidate.model_size_bytes), sizes)
            + 0.25 * _min_max_normalize(candidate.mean_nrmse_std, nrmses)
            + 0.10 * _min_max_normalize(candidate.max_normalized_drift, drifts)
        )
        scores[(candidate.name, candidate.batch_size)] = score
    return scores


def select_deployment_candidate(
    candidates: tuple[DeploymentCandidate, ...] | list[DeploymentCandidate],
    constraints: DeploymentConstraints,
    *,
    policy: str = "lowest_latency",
) -> SelectionResult:
    """Filter candidates by hard constraints, then rank the feasible set."""
    if not candidates:
        raise ValueError("At least one deployment candidate is required.")

    if policy not in _VALID_POLICIES:
        valid = ", ".join(_VALID_POLICIES)
        raise ValueError(f"Unknown selection policy {policy!r}; expected one of: {valid}.")

    typed_policy = cast(SelectionPolicy, policy)
    initial = [evaluate_candidate(candidate, constraints) for candidate in candidates]
    feasible = [decision.candidate for decision in initial if decision.feasible]

    if not feasible:
        return SelectionResult(
            policy=typed_policy,
            constraints=constraints,
            selected_candidate=None,
            decisions=tuple(initial),
        )

    scores: dict[tuple[str, int], float] = {}
    if typed_policy == "balanced":
        scores = _balanced_scores(feasible)

    if typed_policy == "lowest_latency":
        ranked = sorted(
            feasible,
            key=lambda candidate: (
                candidate.median_latency_ms,
                -candidate.mean_r2,
                candidate.model_size_bytes,
                candidate.name,
            ),
        )
    elif typed_policy == "smallest_model":
        ranked = sorted(
            feasible,
            key=lambda candidate: (
                candidate.model_size_bytes,
                candidate.median_latency_ms,
                -candidate.mean_r2,
                candidate.name,
            ),
        )
    elif typed_policy == "highest_accuracy":
        ranked = sorted(
            feasible,
            key=lambda candidate: (
                -candidate.mean_r2,
                candidate.mean_nrmse_std,
                candidate.median_latency_ms,
                candidate.model_size_bytes,
                candidate.name,
            ),
        )
    else:
        ranked = sorted(
            feasible,
            key=lambda candidate: (
                scores[(candidate.name, candidate.batch_size)],
                candidate.median_latency_ms,
                candidate.model_size_bytes,
                -candidate.mean_r2,
                candidate.name,
            ),
        )

    decisions: list[CandidateDecision] = []
    for decision in initial:
        score = scores.get((decision.candidate.name, decision.candidate.batch_size))
        decisions.append(
            CandidateDecision(
                candidate=decision.candidate,
                feasible=decision.feasible,
                rejection_reasons=decision.rejection_reasons,
                score=score,
            )
        )

    return SelectionResult(
        policy=typed_policy,
        constraints=constraints,
        selected_candidate=ranked[0],
        decisions=tuple(decisions),
    )


def _candidate_payload(candidate: DeploymentCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "precision": candidate.precision,
        "provider": candidate.provider,
        "model_path": str(candidate.model_path),
        "model_size_bytes": candidate.model_size_bytes,
        "batch_size": candidate.batch_size,
        "median_latency_ms": candidate.median_latency_ms,
        "mean_r2": candidate.mean_r2,
        "mean_nrmse_std": candidate.mean_nrmse_std,
        "max_normalized_drift": candidate.max_normalized_drift,
        "benchmark_source": str(candidate.benchmark_source),
        "measurement_context": candidate.measurement_context,
    }


def _constraints_payload(constraints: DeploymentConstraints) -> dict[str, Any]:
    return {
        "batch_size": constraints.batch_size,
        "max_latency_ms": constraints.max_latency_ms,
        "max_model_size_mb": constraints.max_model_size_mb,
        "min_r2": constraints.min_r2,
        "max_nrmse_std": constraints.max_nrmse_std,
        "max_normalized_drift": constraints.max_normalized_drift,
        "required_provider": constraints.required_provider,
    }


def selection_result_payload(result: SelectionResult) -> dict[str, Any]:
    """Convert a selection result to a deterministic JSON-ready payload."""
    return {
        "policy": result.policy,
        "constraints": _constraints_payload(result.constraints),
        "selected_candidate": (
            _candidate_payload(result.selected_candidate)
            if result.selected_candidate is not None
            else None
        ),
        "feasible_candidate_count": result.feasible_count,
        "candidate_count": len(result.decisions),
        "decisions": [
            {
                "candidate": _candidate_payload(decision.candidate),
                "feasible": decision.feasible,
                "rejection_reasons": list(decision.rejection_reasons),
                "score": decision.score,
            }
            for decision in result.decisions
        ],
    }


def _markdown_report(result: SelectionResult) -> str:
    lines = [
        "# Neural Deployment Selection",
        "",
        f"- Policy: `{result.policy}`",
        f"- Required batch size: `{result.constraints.batch_size}`",
        f"- Feasible candidates: `{result.feasible_count}/{len(result.decisions)}`",
        "",
    ]

    if result.selected_candidate is None:
        lines.extend(
            [
                "## Selected candidate",
                "",
                "No candidate satisfies all hard deployment constraints.",
                "",
            ]
        )
    else:
        selected = result.selected_candidate
        lines.extend(
            [
                "## Selected candidate",
                "",
                f"- Name: `{selected.name}`",
                f"- Precision: `{selected.precision}`",
                f"- Provider: `{selected.provider}`",
                f"- Batch size: `{selected.batch_size}`",
                f"- Median latency: `{selected.median_latency_ms:.6f} ms`",
                f"- Serialized model size: `{selected.model_size_bytes} bytes`",
                f"- Mean test R2: `{selected.mean_r2:.6f}`",
                f"- Mean test NRMSE: `{selected.mean_nrmse_std:.6f}`",
                f"- Maximum normalized drift: `{selected.max_normalized_drift:.6f}`",
                f"- Measurement context: {selected.measurement_context}",
                "",
            ]
        )

    lines.extend(
        [
            "## Candidate decisions",
            "",
            (
                "| Candidate | Provider | Batch | Latency (ms) | Size (B) | "
                "R2 | NRMSE | Max drift | Feasible |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for decision in result.decisions:
        candidate = decision.candidate
        lines.append(
            "| "
            f"{candidate.name} | {candidate.provider} | {candidate.batch_size} | "
            f"{candidate.median_latency_ms:.6f} | {candidate.model_size_bytes} | "
            f"{candidate.mean_r2:.6f} | {candidate.mean_nrmse_std:.6f} | "
            f"{candidate.max_normalized_drift:.6f} | "
            f"{'yes' if decision.feasible else 'no'} |"
        )

    rejected = [decision for decision in result.decisions if decision.rejection_reasons]
    if rejected:
        lines.extend(["", "## Rejection reasons", ""])
        for decision in rejected:
            lines.append(f"### {decision.candidate.name} / batch {decision.candidate.batch_size}")
            lines.append("")
            for reason in decision.rejection_reasons:
                lines.append(f"- {reason}")
            lines.append("")

    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "Candidates are provider-aware. Cross-provider ranking compares measured deployment "
            "candidates on the tested machine; it must not be described as a precision-only "
            "FP32/FP16/INT8 comparison.",
            "",
            (
                "The model-size field is the serialized canonical ONNX artifact size, "
                "not provider-compiled memory footprint or device memory consumption."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_selection_report(
    result: SelectionResult,
    output_dir: Path = Path("artifacts/deployment_selection"),
) -> DeploymentSelectionArtifacts:
    """Write deterministic JSON and Markdown deployment-selection reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "selection.json"
    markdown_path = output_dir / "selection.md"

    json_path.write_text(
        json.dumps(
            selection_result_payload(result),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown_report(result),
        encoding="utf-8",
    )

    return DeploymentSelectionArtifacts(
        json_path=json_path,
        markdown_path=markdown_path,
    )

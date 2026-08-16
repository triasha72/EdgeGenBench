"""Tests for constraint-aware neural deployment selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgegenbench.deployment.model_selection import (
    DeploymentCandidate,
    DeploymentConstraints,
    evaluate_candidate,
    load_edgegenbench_deployment_candidates,
    select_deployment_candidate,
    selection_result_payload,
    write_selection_report,
)


def _candidate(
    name: str,
    *,
    latency_ms: float = 1.0,
    model_size_bytes: int = 100,
    mean_r2: float = 0.99,
    mean_nrmse_std: float = 0.05,
    drift: float = 0.01,
    batch_size: int = 1,
    provider: str = "CPUExecutionProvider",
) -> DeploymentCandidate:
    return DeploymentCandidate(
        name=name,
        precision=name,
        provider=provider,
        model_path=Path(f"{name}.onnx"),
        model_size_bytes=model_size_bytes,
        batch_size=batch_size,
        median_latency_ms=latency_ms,
        mean_r2=mean_r2,
        mean_nrmse_std=mean_nrmse_std,
        max_normalized_drift=drift,
        benchmark_source=Path("summary.json"),
        measurement_context="unit test",
    )


def test_lowest_latency_policy_selects_fastest_feasible_candidate() -> None:
    result = select_deployment_candidate(
        [
            _candidate("fp32", latency_ms=2.0),
            _candidate("int8", latency_ms=1.0),
        ],
        DeploymentConstraints(batch_size=1),
        policy="lowest_latency",
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.name == "int8"
    assert result.feasible_count == 2


def test_smallest_model_policy_selects_smallest_feasible_candidate() -> None:
    result = select_deployment_candidate(
        [
            _candidate("fp32", model_size_bytes=100),
            _candidate("int8", model_size_bytes=60),
        ],
        DeploymentConstraints(batch_size=1),
        policy="smallest_model",
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.name == "int8"


def test_highest_accuracy_policy_selects_highest_r2_candidate() -> None:
    result = select_deployment_candidate(
        [
            _candidate("fp32", mean_r2=0.997, mean_nrmse_std=0.050),
            _candidate("int8", mean_r2=0.996, mean_nrmse_std=0.051),
        ],
        DeploymentConstraints(batch_size=1),
        policy="highest_accuracy",
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.name == "fp32"


def test_balanced_policy_is_deterministic() -> None:
    candidates = [
        _candidate(
            "accurate",
            latency_ms=2.0,
            model_size_bytes=100,
            mean_r2=0.999,
            mean_nrmse_std=0.02,
            drift=0.0,
        ),
        _candidate(
            "fast",
            latency_ms=1.0,
            model_size_bytes=60,
            mean_r2=0.990,
            mean_nrmse_std=0.08,
            drift=0.03,
        ),
    ]

    first = select_deployment_candidate(
        candidates,
        DeploymentConstraints(batch_size=1),
        policy="balanced",
    )
    second = select_deployment_candidate(
        candidates,
        DeploymentConstraints(batch_size=1),
        policy="balanced",
    )

    assert first.selected_candidate == second.selected_candidate
    assert all(decision.score is not None for decision in first.decisions if decision.feasible)


def test_hard_constraints_record_all_rejection_reasons() -> None:
    decision = evaluate_candidate(
        _candidate(
            "candidate",
            latency_ms=2.0,
            model_size_bytes=2 * 1024 * 1024,
            mean_r2=0.80,
            mean_nrmse_std=0.30,
            drift=0.20,
            batch_size=1,
            provider="CPUExecutionProvider",
        ),
        DeploymentConstraints(
            batch_size=32,
            max_latency_ms=1.0,
            max_model_size_mb=1.0,
            min_r2=0.90,
            max_nrmse_std=0.20,
            max_normalized_drift=0.10,
            required_provider="CoreMLExecutionProvider",
        ),
    )

    assert decision.feasible is False
    assert len(decision.rejection_reasons) == 7


def test_no_feasible_candidate_returns_none_instead_of_hiding_failure() -> None:
    result = select_deployment_candidate(
        [_candidate("slow", latency_ms=2.0)],
        DeploymentConstraints(
            batch_size=1,
            max_latency_ms=1.0,
        ),
    )

    assert result.selected_candidate is None
    assert result.feasible_count == 0


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown selection policy"):
        select_deployment_candidate(
            [_candidate("fp32")],
            DeploymentConstraints(batch_size=1),
            policy="unknown",
        )


def test_report_writer_emits_json_and_markdown(tmp_path: Path) -> None:
    result = select_deployment_candidate(
        [_candidate("fp32")],
        DeploymentConstraints(batch_size=1),
    )

    artifacts = write_selection_report(
        result,
        output_dir=tmp_path,
    )

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert payload["selected_candidate"]["name"] == "fp32"
    assert payload["feasible_candidate_count"] == 1
    assert "# Neural Deployment Selection" in markdown
    assert "provider-aware" in markdown


def test_selection_payload_preserves_rejection_reasons() -> None:
    result = select_deployment_candidate(
        [_candidate("slow", latency_ms=2.0)],
        DeploymentConstraints(
            batch_size=1,
            max_latency_ms=1.0,
        ),
    )

    payload = selection_result_payload(result)

    assert payload["selected_candidate"] is None
    assert payload["decisions"][0]["feasible"] is False
    assert payload["decisions"][0]["rejection_reasons"]


def test_loader_builds_provider_aware_candidates_from_existing_schemas(
    tmp_path: Path,
) -> None:
    fp32_model = tmp_path / "fp32.onnx"
    fp16_model = tmp_path / "fp16.onnx"
    int8_model = tmp_path / "int8.onnx"

    fp32_model.write_bytes(b"x" * 100)
    fp16_model.write_bytes(b"x" * 70)
    int8_model.write_bytes(b"x" * 60)

    training_summary = tmp_path / "neural_training.json"
    fp16_summary = tmp_path / "fp16_summary.json"
    int8_summary = tmp_path / "int8_summary.json"

    training_summary.write_text(
        json.dumps(
            {
                "mean_test_r2": 0.997,
                "mean_test_nrmse_std": 0.050,
            }
        ),
        encoding="utf-8",
    )

    fp16_latency = [
        {
            "batch_size": batch_size,
            "fp32_median_ms": 0.010 * batch_size,
            "fp16_median_ms": 0.008 * batch_size,
        }
        for batch_size in (1, 32, 256)
    ]
    fp16_summary.write_text(
        json.dumps(
            {
                "provider": "CoreMLExecutionProvider",
                "accuracy": {
                    "fp16_mean_r2": 0.996,
                    "fp16_mean_nrmse_std": 0.051,
                },
                "model_size": {
                    "fp32_model_size_bytes": 100,
                    "fp16_model_size_bytes": 70,
                },
                "provider_drift": {
                    "fp32_cpu_vs_fp32_coreml_max_normalized_abs": 0.0001,
                },
                "precision_drift": {
                    "fp32_coreml_vs_fp16_coreml_max_normalized_abs": 0.005,
                },
                "latency": {
                    "summary": fp16_latency,
                },
            }
        ),
        encoding="utf-8",
    )

    int8_latency = [
        {
            "batch_size": batch_size,
            "fp32_median_ms": 0.010 * batch_size,
            "int8_median_ms": 0.007 * batch_size,
        }
        for batch_size in (1, 32, 256)
    ]
    int8_summary.write_text(
        json.dumps(
            {
                "provider": "CPUExecutionProvider",
                "predictive_quality": {
                    "int8_mean_r2": 0.995,
                    "int8_mean_nrmse_std": 0.052,
                },
                "model_size": {
                    "fp32_bytes": 100,
                    "int8_bytes": 60,
                },
                "drift": {
                    "max_normalized_absolute": 0.050,
                },
                "latency": {
                    "summary": int8_latency,
                },
            }
        ),
        encoding="utf-8",
    )

    candidates = load_edgegenbench_deployment_candidates(
        neural_training_summary_path=training_summary,
        fp16_benchmark_summary_path=fp16_summary,
        int8_benchmark_summary_path=int8_summary,
        fp32_model_path=fp32_model,
        fp16_model_path=fp16_model,
        int8_model_path=int8_model,
    )

    assert len(candidates) == 12
    assert {candidate.name for candidate in candidates} == {
        "fp32_cpu",
        "mixed_int8_fp32_cpu",
        "fp32_coreml",
        "fp16_coreml",
    }

    batch_256_int8 = next(
        candidate
        for candidate in candidates
        if candidate.name == "mixed_int8_fp32_cpu" and candidate.batch_size == 256
    )

    assert batch_256_int8.provider == "CPUExecutionProvider"
    assert batch_256_int8.model_size_bytes == 60
    assert batch_256_int8.median_latency_ms == pytest.approx(1.792)
    assert batch_256_int8.max_normalized_drift == pytest.approx(0.050)


def test_loader_rejects_model_size_metadata_mismatch(tmp_path: Path) -> None:
    fp32_model = tmp_path / "fp32.onnx"
    fp16_model = tmp_path / "fp16.onnx"
    int8_model = tmp_path / "int8.onnx"
    fp32_model.write_bytes(b"x" * 100)
    fp16_model.write_bytes(b"x" * 70)
    int8_model.write_bytes(b"x" * 60)

    training_summary = tmp_path / "training.json"
    fp16_summary = tmp_path / "fp16.json"
    int8_summary = tmp_path / "int8.json"

    training_summary.write_text(
        json.dumps(
            {
                "mean_test_r2": 0.997,
                "mean_test_nrmse_std": 0.050,
            }
        ),
        encoding="utf-8",
    )
    fp16_summary.write_text(
        json.dumps(
            {
                "provider": "CoreMLExecutionProvider",
                "accuracy": {
                    "fp16_mean_r2": 0.996,
                    "fp16_mean_nrmse_std": 0.051,
                },
                "model_size": {
                    "fp32_model_size_bytes": 101,
                    "fp16_model_size_bytes": 70,
                },
                "provider_drift": {
                    "fp32_cpu_vs_fp32_coreml_max_normalized_abs": 0.0001,
                },
                "precision_drift": {
                    "fp32_coreml_vs_fp16_coreml_max_normalized_abs": 0.005,
                },
                "latency": {
                    "summary": [
                        {
                            "batch_size": 1,
                            "fp32_median_ms": 0.01,
                            "fp16_median_ms": 0.008,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    int8_summary.write_text(
        json.dumps(
            {
                "provider": "CPUExecutionProvider",
                "predictive_quality": {
                    "int8_mean_r2": 0.995,
                    "int8_mean_nrmse_std": 0.052,
                },
                "model_size": {
                    "fp32_bytes": 100,
                    "int8_bytes": 60,
                },
                "drift": {
                    "max_normalized_absolute": 0.050,
                },
                "latency": {
                    "summary": [
                        {
                            "batch_size": 1,
                            "fp32_median_ms": 0.01,
                            "int8_median_ms": 0.007,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model-size metadata disagrees"):
        load_edgegenbench_deployment_candidates(
            neural_training_summary_path=training_summary,
            fp16_benchmark_summary_path=fp16_summary,
            int8_benchmark_summary_path=int8_summary,
            fp32_model_path=fp32_model,
            fp16_model_path=fp16_model,
            int8_model_path=int8_model,
        )

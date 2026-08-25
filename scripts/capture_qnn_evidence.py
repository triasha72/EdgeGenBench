#!/usr/bin/env python3
"""Assemble and validate physical-device QNN evidence from retained run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from build_release_evidence import validate_qnn_evidence_bundle


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_qnn_evidence(
    *,
    benchmark_path: Path,
    context_binary: Path,
    placement_report: Path,
    profile: Path,
    logcat: Path,
    model: Path,
    input_data: Path,
    output_dir: Path,
    ort_version: str,
    qairt_version: str,
    device_fingerprint: str,
    soc_model: str,
    cold_ms: float,
    peak_rss_mb: float,
    max_abs_drift_vs_fp32: float,
    max_allowed_abs_drift: float,
    power_status: str = "not measured",
    power_tool: str | None = None,
) -> Path:
    benchmark = _load_json(benchmark_path)
    benchmark_placement = benchmark.get("placement")
    if benchmark.get("backend") != "qnn" or not isinstance(benchmark_placement, dict):
        raise ValueError("benchmark must be a QNN native runtime result")
    if (
        benchmark_placement.get("backend") != "QNNExecutionProvider"
        or benchmark_placement.get("cpu_fallback") is not False
        or benchmark_placement.get("hardware_measurement") is not True
    ):
        raise ValueError("benchmark does not prove fail-closed QNN session configuration")
    latency = benchmark.get("latency_ms")
    if not isinstance(latency, dict):
        raise ValueError("benchmark requires latency_ms")
    batch_size = benchmark.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("benchmark requires a positive batch_size")

    placement = _load_json(placement_report)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    sources = {
        "context_binary": context_binary,
        "placement_report": placement_report,
        "profile": profile,
        "logcat": logcat,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise ValueError(f"required QNN artifact is missing: {source}")
        target = artifacts_dir / source.name
        shutil.copy2(source, target)
        artifacts[name] = {
            "path": target.relative_to(output_dir).as_posix(),
            "sha256": _sha256(target),
        }

    p50 = float(latency["p50"])
    p95 = float(latency["p95"])
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "project": "EdgeGenBench",
        "backend": "QNNExecutionProvider",
        "cpu_fallback": False,
        "identity": {
            "model_sha256": _sha256(model),
            "input_sha256": _sha256(input_data),
            "ort_version": ort_version,
            "qairt_version": qairt_version,
            "device_fingerprint": device_fingerprint,
            "soc_model": soc_model,
        },
        "placement": {
            "provider": placement.get("provider"),
            "qnn_node_count": placement.get("qnn_node_count"),
            "cpu_node_count": placement.get("cpu_node_count"),
            "unassigned_node_count": placement.get("unassigned_node_count"),
        },
        "artifacts": artifacts,
        "measurements": {
            "cold_ms": cold_ms,
            "warm_p50_ms": p50,
            "warm_p95_ms": p95,
            "throughput_per_second": batch_size * 1000.0 / p50,
            "peak_rss_mb": peak_rss_mb,
            "max_abs_drift_vs_fp32": max_abs_drift_vs_fp32,
        },
        "max_allowed_abs_drift": max_allowed_abs_drift,
        "measurement_claims": {
            "npu_placement": "validated",
            "power": power_status,
        },
    }
    if power_tool:
        bundle["measurement_claims"]["power_tool"] = power_tool
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    validate_qnn_evidence_bundle(evidence_path)
    return evidence_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--context-binary", type=Path, required=True)
    parser.add_argument("--placement-report", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--logcat", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", dest="input_data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ort-version", required=True)
    parser.add_argument("--qairt-version", required=True)
    parser.add_argument("--device-fingerprint", required=True)
    parser.add_argument("--soc-model", required=True)
    parser.add_argument("--cold-ms", type=float, required=True)
    parser.add_argument("--peak-rss-mb", type=float, required=True)
    parser.add_argument("--max-abs-drift-vs-fp32", type=float, required=True)
    parser.add_argument("--max-allowed-abs-drift", type=float, required=True)
    parser.add_argument("--power-status", default="not measured")
    parser.add_argument("--power-tool")
    args = parser.parse_args()
    path = capture_qnn_evidence(**vars(args))
    print(f"Validated QNN evidence captured: {path}")


if __name__ == "__main__":
    main()

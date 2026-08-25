import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_qnn_evidence.py"
CaptureQnnEvidence = Callable[..., Path]
capture_qnn_evidence = cast(CaptureQnnEvidence, runpy.run_path(SCRIPT)["capture_qnn_evidence"])


def _file(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def _capture(tmp_path: Path, *, cpu_fallback: bool = False) -> Path:
    benchmark = {
        "schema_version": 1,
        "backend": "qnn",
        "batch_size": 1,
        "latency_ms": {"mean": 0.6, "p50": 0.5, "p95": 0.7},
        "placement": {
            "backend": "QNNExecutionProvider",
            "cpu_fallback": cpu_fallback,
            "context_cache": True,
            "hardware_measurement": True,
        },
    }
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    placement = _file(
        tmp_path / "placement.json",
        json.dumps(
            {
                "provider": "QNNExecutionProvider",
                "qnn_node_count": 9,
                "cpu_node_count": 0,
                "unassigned_node_count": 0,
            }
        ),
    )
    return capture_qnn_evidence(
        benchmark_path=benchmark_path,
        context_binary=_file(tmp_path / "context.bin", "context"),
        placement_report=placement,
        profile=_file(tmp_path / "profile.json", "profile"),
        logcat=_file(tmp_path / "logcat.txt", "QNNExecutionProvider"),
        model=_file(tmp_path / "model.onnx", "model"),
        input_data=_file(tmp_path / "input.bin", "input"),
        output_dir=tmp_path / "evidence",
        ort_version="1.22.0",
        qairt_version="2.45.0",
        device_fingerprint="vendor/device/build",
        soc_model="Snapdragon test SoC",
        cold_ms=2.0,
        peak_rss_mb=40.0,
        max_abs_drift_vs_fp32=1e-5,
        max_allowed_abs_drift=1e-4,
    )


def test_capture_builds_self_validating_qnn_bundle(tmp_path: Path) -> None:
    evidence = _capture(tmp_path)
    value = json.loads(evidence.read_text(encoding="utf-8"))
    assert value["backend"] == "QNNExecutionProvider"
    assert value["measurements"]["throughput_per_second"] == 2000.0
    assert len(value["artifacts"]) == 4


def test_capture_rejects_benchmark_with_cpu_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fail-closed QNN"):
        _capture(tmp_path, cpu_fallback=True)

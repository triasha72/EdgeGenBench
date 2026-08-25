import json
from pathlib import Path

import pytest

from scripts.build_release_evidence import build_release_evidence


def _benchmark(path: Path, preprocess: str, output: float = 0.15616) -> Path:
    value = {
        "schema_version": 1,
        "backend": "reference",
        "preprocess": preprocess,
        "warmup_runs": 5,
        "batch_size": 1,
        "measured_runs": 100,
        "latency_ms": {"mean": 0.01, "p50": 0.009, "p95": 0.012},
        "output": output,
        "placement": {
            "backend": "reference",
            "operators": {"ReferenceLinear": 1},
            "cpu_fallback": False,
            "hardware_measurement": False,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    apk = tmp_path / "app.apk"
    apk.write_bytes(b"valid-test-apk")
    alignment = tmp_path / "alignment.txt"
    alignment.write_text("APK ZIP and ELF 16 KiB alignment checks passed\n", encoding="utf-8")
    return (
        _benchmark(tmp_path / "baseline.json", "baseline"),
        _benchmark(tmp_path / "fused.json", "fused"),
        apk,
        alignment,
    )


def test_builds_hashed_cross_runtime_manifest(tmp_path: Path) -> None:
    manifest_path = build_release_evidence(
        *_inputs(tmp_path), tmp_path / "release", git_revision="abc123", version="0.1.5"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["acceptance"]["baseline_fused_max_abs_drift"] == 0
    assert manifest["acceptance"]["android_16kb_compatible"] is True
    assert manifest["acceptance"]["qnn_npu_placement"] == "not tested in CI"
    assert manifest["device_evidence"]["status"] == "not_supplied"
    assert {item["path"] for item in manifest["files"]} == {
        "android/16kb-alignment.txt",
        "android/app.apk",
        "native/baseline.json",
        "native/fused.json",
    }
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_rejects_preprocessing_output_drift(tmp_path: Path) -> None:
    baseline, _, apk, alignment = _inputs(tmp_path)
    fused = _benchmark(tmp_path / "fused-drift.json", "fused", output=0.2)
    with pytest.raises(ValueError, match="output drift"):
        build_release_evidence(
            baseline,
            fused,
            apk,
            alignment,
            tmp_path / "release",
            git_revision="abc123",
            version="0.1.5",
        )


def test_rejects_false_hardware_measurement_claim(tmp_path: Path) -> None:
    baseline, fused, apk, alignment = _inputs(tmp_path)
    value = json.loads(fused.read_text(encoding="utf-8"))
    value["placement"]["hardware_measurement"] = True
    fused.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be presented"):
        build_release_evidence(
            baseline,
            fused,
            apk,
            alignment,
            tmp_path / "release",
            git_revision="abc123",
            version="0.1.5",
        )

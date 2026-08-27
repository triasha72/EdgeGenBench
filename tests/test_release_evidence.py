import hashlib
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

BuildReleaseEvidence = Callable[..., Path]
SCRIPT = Path(__file__).parents[1] / "scripts" / "build_release_evidence.py"
RELEASE_FUNCTIONS = runpy.run_path(SCRIPT)
build_release_evidence = cast(BuildReleaseEvidence, RELEASE_FUNCTIONS["build_release_evidence"])
validate_android_device_bundle = cast(
    Callable[..., dict[str, object]], RELEASE_FUNCTIONS["validate_android_device_bundle"]
)
validate_qnn_evidence_bundle = cast(
    Callable[..., dict[str, object]], RELEASE_FUNCTIONS["validate_qnn_evidence_bundle"]
)


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


def _device_bundle(path: Path, *, output_drift: float = 0.0) -> Path:
    result = {
        "schema_version": 1,
        "backend": "reference",
        "cpu_fallback": False,
        "cold_ms": 0.01,
        "warm_mean_ms": 0.004,
        "warm_p95_ms": 0.005,
        "runs": 100,
        "baseline_preprocess_mean_ms": 0.2,
        "fused_preprocess_mean_ms": 0.1,
        "preprocess_speedup_x": 2.0,
        "preprocess_max_abs_drift": 0.0,
        "output_max_abs_drift": output_drift,
        "output": 0.15616,
        "runtime_page_size_bytes": 4096,
    }
    bundle = {
        "schema_version": 1,
        "project": "EdgeGenBench",
        "app": {"version_name": "0.1.7", "version_code": 8, "git_revision": "abc123"},
        "device": {
            "manufacturer": "Samsung",
            "model": "SM-A356E",
            "android_release": "16",
            "sdk_int": 36,
            "supported_abis": ["arm64-v8a"],
        },
        "measurement_claims": {
            "backend": "reference",
            "qnn_npu_placement": "not tested",
            "power": "not measured",
            "thermal": "not included in app export",
        },
        "result_count": 3,
        "results": [result, {**result, "cold_ms": 0.02}, {**result, "cold_ms": 0.03}],
    }
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_validates_and_summarizes_android_export(tmp_path: Path) -> None:
    summary = validate_android_device_bundle(_device_bundle(tmp_path / "device.json"))
    assert summary["status"] == "validated_reference"
    assert summary["result_count"] == 3
    assert summary["runtime_page_sizes_bytes"] == [4096]
    metrics = cast(dict[str, object], summary["metrics"])
    cold = cast(dict[str, float], metrics["cold_ms"])
    assert cold["mean"] == pytest.approx(0.02)


def test_rejects_android_export_with_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="drift gate"):
        validate_android_device_bundle(_device_bundle(tmp_path / "device.json", output_drift=0.01))


def test_release_bundle_marks_validated_device_evidence(tmp_path: Path) -> None:
    manifest_path = build_release_evidence(
        *_inputs(tmp_path),
        tmp_path / "release",
        git_revision="abc123",
        version="0.1.7",
        device_evidence=_device_bundle(tmp_path / "device.json"),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["device_evidence"]["status"] == "validated_reference"
    assert (tmp_path / "release/device/report.md").is_file()
    assert (tmp_path / "release/device/summary.json").is_file()


def _qnn_bundle(tmp_path: Path, *, cpu_fallback: bool = False, drift: float = 1e-5) -> Path:
    artifacts = {}
    for name, filename in {
        "context_binary": "qnn_context.bin",
        "placement_report": "placement.json",
        "profile": "profile.json",
        "logcat": "logcat.txt",
    }.items():
        artifact = tmp_path / filename
        artifact.write_bytes(f"test-{name}".encode())
        artifacts[name] = {
            "path": filename,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    bundle = {
        "schema_version": 1,
        "project": "EdgeGenBench",
        "backend": "QNNExecutionProvider",
        "cpu_fallback": cpu_fallback,
        "identity": {
            "model_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "ort_version": "1.22.0",
            "qairt_version": "2.45.0",
            "device_fingerprint": "vendor/device/build",
            "soc_model": "Snapdragon test SoC",
        },
        "placement": {
            "provider": "QNNExecutionProvider",
            "qnn_node_count": 9,
            "cpu_node_count": 0,
            "unassigned_node_count": 0,
        },
        "artifacts": artifacts,
        "measurements": {
            "cold_ms": 2.0,
            "warm_p50_ms": 0.5,
            "warm_p95_ms": 0.7,
            "throughput_per_second": 2000.0,
            "peak_rss_mb": 40.0,
            "max_abs_drift_vs_fp32": drift,
        },
        "max_allowed_abs_drift": 1e-4,
        "measurement_claims": {"npu_placement": "validated", "power": "not measured"},
    }
    path = tmp_path / "qnn-evidence.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_validates_qnn_evidence_and_artifact_checksums(tmp_path: Path) -> None:
    summary = validate_qnn_evidence_bundle(_qnn_bundle(tmp_path))
    assert summary["status"] == "validated_qnn_npu"
    assert summary["cpu_fallback"] is False
    assert len(cast(dict[str, object], summary["verified_artifacts"])) == 4


def test_rejects_qnn_cpu_fallback_and_excess_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disable CPU fallback"):
        validate_qnn_evidence_bundle(_qnn_bundle(tmp_path, cpu_fallback=True))
    with pytest.raises(ValueError, match="drift exceeds"):
        validate_qnn_evidence_bundle(_qnn_bundle(tmp_path, drift=0.01))


def test_release_bundle_retains_validated_qnn_artifacts(tmp_path: Path) -> None:
    manifest_path = build_release_evidence(
        *_inputs(tmp_path),
        tmp_path / "release",
        git_revision="abc123",
        version="0.1.8",
        qnn_evidence=_qnn_bundle(tmp_path),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["qnn_evidence"]["status"] == "validated_qnn_npu"
    assert (tmp_path / "release/qnn/summary.json").is_file()
    assert (tmp_path / "release/qnn/artifacts/qnn_context.bin").is_file()


def test_release_bundle_retains_ios_simulator_acceptance(tmp_path: Path) -> None:
    ios = tmp_path / "ios-evidence"
    ios.mkdir()
    for name in (
        "EdgeGenBench-ios-simulator-app.zip",
        "ios-tests.xcresult.zip",
        "xcode-version.txt",
        "checksums.txt",
    ):
        (ios / name).write_bytes(name.encode())
    manifest_path = build_release_evidence(
        *_inputs(tmp_path),
        tmp_path / "release",
        git_revision="abc123",
        version="0.1.8",
        ios_simulator_evidence=ios,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["acceptance"]["ios_coreml_simulator_build_and_tests"] is True
    assert manifest["ios_simulator_evidence"]["status"] == "validated_in_ci"
    assert (tmp_path / "release/ios-simulator/ios-tests.xcresult.zip").is_file()

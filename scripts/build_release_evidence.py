#!/usr/bin/env python3
"""Validate and package cross-runtime EdgeGenBench release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def validate_android_device_bundle(path: Path, *, min_results: int = 3) -> dict[str, Any]:
    bundle = _load_json(path)
    if bundle.get("schema_version") != 1 or bundle.get("project") != "EdgeGenBench":
        raise ValueError("unsupported Android evidence bundle")
    app = bundle.get("app")
    device = bundle.get("device")
    claims = bundle.get("measurement_claims")
    results = bundle.get("results")
    if not isinstance(app, dict) or not all(
        app.get(name) for name in ("version_name", "version_code", "git_revision")
    ):
        raise ValueError("Android evidence requires app version and Git revision")
    if not isinstance(device, dict) or not all(
        device.get(name)
        for name in (
            "manufacturer",
            "model",
            "android_release",
            "sdk_int",
            "supported_abis",
        )
    ):
        raise ValueError("Android evidence requires complete device identity")
    if not isinstance(claims, dict):
        raise ValueError("Android evidence requires measurement claims")
    if claims.get("backend") != "reference" or claims.get("qnn_npu_placement") != "not tested":
        raise ValueError("reference evidence must not claim QNN/NPU placement")
    if claims.get("power") != "not measured":
        raise ValueError("power claims require a separate measured-power evidence path")
    if claims.get("thermal") != "not included in app export":
        raise ValueError("app export must not claim thermal evidence")
    if not isinstance(results, list) or len(results) < min_results:
        raise ValueError(f"Android evidence requires at least {min_results} results")
    if bundle.get("result_count") != len(results):
        raise ValueError("Android evidence result_count does not match results")

    numeric_fields = (
        "cold_ms",
        "warm_mean_ms",
        "warm_p95_ms",
        "baseline_preprocess_mean_ms",
        "fused_preprocess_mean_ms",
        "preprocess_speedup_x",
        "preprocess_max_abs_drift",
        "output_max_abs_drift",
        "output",
    )
    for index, result in enumerate(results):
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise ValueError(f"result {index} has an unsupported schema")
        if result.get("backend") != "reference" or result.get("cpu_fallback") is not False:
            raise ValueError(f"result {index} has invalid reference placement metadata")
        for name in numeric_fields:
            value = result.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"result {index} field {name} must be finite and numeric")
        if any(float(result[name]) < 0 for name in numeric_fields if name != "output"):
            raise ValueError(f"result {index} contains a negative measurement")
        if float(result["warm_mean_ms"]) <= 0:
            raise ValueError(f"result {index} requires positive warm latency")
        if (
            max(
                float(result["preprocess_max_abs_drift"]),
                float(result["output_max_abs_drift"]),
            )
            > 1e-6
        ):
            raise ValueError(f"result {index} fails the numerical drift gate")
        if (
            not isinstance(result.get("runtime_page_size_bytes"), int)
            or result["runtime_page_size_bytes"] <= 0
        ):
            raise ValueError(f"result {index} requires a runtime page size")

    def metric(name: str) -> dict[str, float]:
        values = [float(result[name]) for result in results]
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    page_sizes = sorted({int(result["runtime_page_size_bytes"]) for result in results})
    outputs = [float(result["output"]) for result in results]
    summary = {
        "schema_version": 1,
        "status": "validated_reference",
        "app": app,
        "device": device,
        "measurement_claims": claims,
        "result_count": len(results),
        "runtime_page_sizes_bytes": page_sizes,
        "metrics": {
            "cold_ms": metric("cold_ms"),
            "warm_mean_ms": metric("warm_mean_ms"),
            "warm_p95_ms": metric("warm_p95_ms"),
            "preprocess_speedup_x": metric("preprocess_speedup_x"),
            "throughput_inferences_per_second": {
                "mean": statistics.fmean(1000.0 / float(r["warm_mean_ms"]) for r in results)
            },
            "max_preprocess_abs_drift": max(float(r["preprocess_max_abs_drift"]) for r in results),
            "max_output_abs_drift": max(float(r["output_max_abs_drift"]) for r in results),
            "max_output_spread": max(outputs) - min(outputs),
        },
    }
    return summary


def write_android_device_report(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["metrics"]
    app = summary["app"]
    device = summary["device"]
    claims = summary["measurement_claims"]
    lines = [
        "# EdgeGenBench Android device evidence",
        "",
        f"- Status: `{summary['status']}`",
        f"- App: `{app['version_name']}` (`{app['git_revision']}`)",
        f"- Device: {device['manufacturer']} {device['model']}",
        f"- Android: {device['android_release']} / API {device['sdk_int']}",
        f"- Retained runs: {summary['result_count']}",
        f"- Runtime page size(s): {summary['runtime_page_sizes_bytes']}",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("cold_ms", "warm_mean_ms", "warm_p95_ms", "preprocess_speedup_x"):
        value = metrics[name]
        lines.append(
            f"| {name} | {value['mean']:.6f} | {value['median']:.6f} | "
            f"{value['min']:.6f} | {value['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Maximum preprocessing drift: `{metrics['max_preprocess_abs_drift']:.9g}`",
            f"- Maximum output drift: `{metrics['max_output_abs_drift']:.9g}`",
            f"- Output spread across runs: `{metrics['max_output_spread']:.9g}`",
            f"- Backend: `{claims['backend']}`",
            f"- QNN/NPU placement: `{claims['qnn_npu_placement']}`",
            f"- Power: `{claims['power']}`",
            f"- Thermal: `{claims['thermal']}`",
            "",
            "This report validates the exported reference-backend contract. It is not QNN/NPU, "
            "measured-power, or thermal evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_qnn_evidence_bundle(path: Path) -> dict[str, Any]:
    """Validate a physical-device QNN evidence manifest and its retained files."""
    bundle = _load_json(path)
    if bundle.get("schema_version") != 1 or bundle.get("project") != "EdgeGenBench":
        raise ValueError("unsupported QNN evidence bundle")
    if bundle.get("backend") != "QNNExecutionProvider":
        raise ValueError("QNN evidence must name QNNExecutionProvider")
    if bundle.get("cpu_fallback") is not False:
        raise ValueError("QNN evidence must disable CPU fallback")

    identity = bundle.get("identity")
    if not isinstance(identity, dict) or not all(
        identity.get(name)
        for name in (
            "model_sha256",
            "input_sha256",
            "ort_version",
            "qairt_version",
            "device_fingerprint",
            "soc_model",
        )
    ):
        raise ValueError("QNN evidence requires complete model, runtime, and device identity")
    for name in ("model_sha256", "input_sha256"):
        value = identity[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"QNN identity {name} must be a SHA-256 digest")

    placement = bundle.get("placement")
    if not isinstance(placement, dict):
        raise ValueError("QNN evidence requires placement metadata")
    if (
        placement.get("provider") != "QNNExecutionProvider"
        or placement.get("unassigned_node_count") != 0
        or placement.get("cpu_node_count") != 0
        or not isinstance(placement.get("qnn_node_count"), int)
        or placement["qnn_node_count"] <= 0
    ):
        raise ValueError("QNN evidence does not prove exclusive QNN placement")

    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("QNN evidence requires retained artifacts")
    root = path.parent
    verified_artifacts: dict[str, dict[str, Any]] = {}
    for name in ("context_binary", "placement_report", "profile", "logcat"):
        item = artifacts.get(name)
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"QNN evidence requires {name} path and SHA-256")
        artifact_path = root / str(item["path"])
        if not artifact_path.is_file() or _sha256(artifact_path) != item["sha256"]:
            raise ValueError(f"QNN artifact {name} is missing or fails checksum validation")
        verified_artifacts[name] = {
            "path": str(item["path"]),
            "bytes": artifact_path.stat().st_size,
            "sha256": item["sha256"],
        }

    measurements = bundle.get("measurements")
    required = ("cold_ms", "warm_p50_ms", "warm_p95_ms", "throughput_per_second", "peak_rss_mb")
    if not isinstance(measurements, dict):
        raise ValueError("QNN evidence requires measurements")
    for name in required:
        value = measurements.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"QNN measurement {name} must be finite and positive")
    if float(measurements["warm_p95_ms"]) < float(measurements["warm_p50_ms"]):
        raise ValueError("QNN warm p95 latency cannot be lower than p50")
    drift = measurements.get("max_abs_drift_vs_fp32")
    if not isinstance(drift, (int, float)) or not math.isfinite(float(drift)) or drift < 0:
        raise ValueError("QNN output drift must be finite and non-negative")
    drift_limit = bundle.get("max_allowed_abs_drift")
    if not isinstance(drift_limit, (int, float)) or drift_limit < 0 or drift > drift_limit:
        raise ValueError("QNN output drift exceeds the declared acceptance limit")

    claims = bundle.get("measurement_claims")
    if not isinstance(claims, dict) or claims.get("npu_placement") != "validated":
        raise ValueError("QNN evidence must explicitly claim validated NPU placement")
    power = claims.get("power")
    if power != "not measured" and not claims.get("power_tool"):
        raise ValueError("measured power claims require a named measurement tool")

    return {
        "schema_version": 1,
        "status": "validated_qnn_npu",
        "backend": "QNNExecutionProvider",
        "cpu_fallback": False,
        "identity": identity,
        "placement": placement,
        "measurements": measurements,
        "measurement_claims": claims,
        "verified_artifacts": verified_artifacts,
    }


def _validate_benchmark(result: dict[str, Any], expected_preprocess: str) -> None:
    if result.get("schema_version") != 1:
        raise ValueError("native benchmark schema_version must be 1")
    if result.get("backend") != "reference":
        raise ValueError("release baseline must identify the reference backend")
    if result.get("preprocess") != expected_preprocess:
        raise ValueError(f"expected {expected_preprocess} preprocessing result")
    if not isinstance(result.get("output"), (int, float)):
        raise ValueError("native benchmark output must be numeric")
    latency = result.get("latency_ms")
    if not isinstance(latency, dict) or any(
        not isinstance(latency.get(name), (int, float)) or latency[name] < 0
        for name in ("mean", "p50", "p95")
    ):
        raise ValueError("native benchmark latency fields must be non-negative numbers")
    placement = result.get("placement")
    if not isinstance(placement, dict):
        raise ValueError("native benchmark placement must be an object")
    if (
        placement.get("backend") != "reference"
        or placement.get("hardware_measurement") is not False
    ):
        raise ValueError("reference results must not be presented as hardware measurements")


def build_release_evidence(
    baseline_path: Path,
    fused_path: Path,
    apk_path: Path,
    alignment_path: Path,
    output_dir: Path,
    *,
    git_revision: str,
    version: str,
    device_evidence: Path | None = None,
    qnn_evidence: Path | None = None,
) -> Path:
    baseline = _load_json(baseline_path)
    fused = _load_json(fused_path)
    _validate_benchmark(baseline, "baseline")
    _validate_benchmark(fused, "fused")
    drift = abs(float(baseline["output"]) - float(fused["output"]))
    if drift > 1e-6:
        raise ValueError(f"baseline/fused output drift {drift} exceeds 1e-6")
    if apk_path.suffix != ".apk" or not apk_path.is_file():
        raise ValueError("an Android APK is required")
    alignment = alignment_path.read_text(encoding="utf-8")
    if "APK ZIP and ELF 16 KiB alignment checks passed" not in alignment:
        raise ValueError("Android 16 KiB compatibility evidence did not pass")

    output_dir.mkdir(parents=True, exist_ok=True)
    native_dir = output_dir / "native"
    android_dir = output_dir / "android"
    native_dir.mkdir(exist_ok=True)
    android_dir.mkdir(exist_ok=True)
    copied = {
        "native/baseline.json": baseline_path,
        "native/fused.json": fused_path,
        f"android/{apk_path.name}": apk_path,
        "android/16kb-alignment.txt": alignment_path,
    }
    for relative, source in copied.items():
        shutil.copy2(source, output_dir / relative)

    device_status: dict[str, Any] = {
        "status": "not_supplied",
        "claim": "No physical-device, NPU-placement, thermal, or power claim is made by CI.",
    }
    if device_evidence is not None:
        destination = output_dir / "device"
        if device_evidence.is_file():
            destination.mkdir(exist_ok=True)
            summary = validate_android_device_bundle(device_evidence)
            shutil.copy2(device_evidence, destination / "android-evidence.json")
            (destination / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            write_android_device_report(summary, destination / "report.md")
            device_status = {
                "status": "validated_reference",
                "path": "device/android-evidence.json",
                "report": "device/report.md",
                "claim": "Validated reference-backend device evidence; not QNN or power evidence.",
            }
        elif device_evidence.is_dir():
            shutil.copytree(device_evidence, destination, dirs_exist_ok=True)
            device_status = {
                "status": "supplied_unverified",
                "path": "device",
                "claim": "Device evidence is retained verbatim and requires reviewer verification.",
            }
        else:
            raise ValueError("device evidence must be a JSON file or directory")

    qnn_status: dict[str, Any] = {
        "status": "not_supplied",
        "claim": "No QNN/NPU placement claim is made by this release bundle.",
    }
    if qnn_evidence is not None:
        summary = validate_qnn_evidence_bundle(qnn_evidence)
        destination = output_dir / "qnn"
        destination.mkdir(exist_ok=True)
        shutil.copy2(qnn_evidence, destination / "evidence.json")
        for item in summary["verified_artifacts"].values():
            source = qnn_evidence.parent / item["path"]
            target = destination / "artifacts" / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        qnn_status = {
            "status": "validated_qnn_npu",
            "path": "qnn/evidence.json",
            "summary": "qnn/summary.json",
            "claim": "Exclusive QNN placement validated with CPU fallback disabled.",
        }

    files = []
    for path in sorted(
        p for p in output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"
    ):
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "project": "EdgeGenBench",
        "version": version,
        "git_revision": git_revision,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "acceptance": {
            "native_baseline_passed": True,
            "native_fused_passed": True,
            "baseline_fused_max_abs_drift": drift,
            "android_16kb_compatible": True,
            "cpu_fallback_claim": "not applicable to deterministic reference backend",
            "qnn_npu_placement": "not tested in CI",
            "power": "not measured",
        },
        "device_evidence": device_status,
        "qnn_evidence": qnn_status,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--device-evidence", type=Path)
    parser.add_argument("--qnn-evidence", type=Path)
    args = parser.parse_args()
    manifest = build_release_evidence(
        args.baseline,
        args.fused,
        args.apk,
        args.alignment_report,
        args.output_dir,
        git_revision=args.git_revision,
        version=args.version,
        device_evidence=args.device_evidence,
        qnn_evidence=args.qnn_evidence,
    )
    print(f"Release evidence validated: {manifest}")


if __name__ == "__main__":
    main()

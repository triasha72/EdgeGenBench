#!/usr/bin/env python3
"""Validate and package cross-runtime EdgeGenBench release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
        if not device_evidence.is_dir():
            raise ValueError("device evidence must be a directory")
        destination = output_dir / "device"
        shutil.copytree(device_evidence, destination, dirs_exist_ok=True)
        device_status = {
            "status": "supplied_unverified",
            "path": "device",
            "claim": "Device evidence is retained verbatim and requires reviewer verification.",
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
    )
    print(f"Release evidence validated: {manifest}")


if __name__ == "__main__":
    main()

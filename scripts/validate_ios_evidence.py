#!/usr/bin/env python3
"""Validate an EdgeGenBench iPhone Core ML evidence export and write a report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ios_evidence(
    evidence_path: Path,
    *,
    model_path: Path,
    preprocessing_path: Path,
    allow_simulator: bool = False,
) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or evidence.get("schemaVersion") != "1.0":
        raise ValueError("unsupported iOS evidence schema")
    if evidence.get("backend") != "CoreML" or evidence.get("requestedComputeUnits") != "all":
        raise ValueError("evidence must identify Core ML with requested compute units")
    if evidence.get("neuralEnginePlacement") != "not_measured":
        raise ValueError("ANE placement cannot be inferred without retained placement evidence")
    if evidence.get("powerMeasurement") != "not_measured":
        raise ValueError("power claims require a named calibrated measurement tool")

    device = evidence.get("device")
    latency = evidence.get("latency")
    if not isinstance(device, dict) or not isinstance(latency, dict):
        raise ValueError("device identity and latency summary are required")
    if bool(device.get("simulator")) and not allow_simulator:
        raise ValueError("physical-iPhone evidence cannot come from a simulator")
    if device.get("systemName") != "iOS" and not allow_simulator:
        raise ValueError("physical evidence must identify iOS")
    if int(latency.get("warmRuns", 0)) < 100:
        raise ValueError("at least 100 warm inference runs are required")
    for name in ("coldMs", "warmMeanMs", "warmP95Ms"):
        if float(latency.get(name, 0)) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(evidence.get("outputMaxAbsDrift", 1.0)) > 1e-6:
        raise ValueError("iOS repeated-output drift exceeds tolerance")
    if evidence.get("sourceModelSha256") != _sha256(model_path):
        raise ValueError("iOS source-model provenance does not match the repository")
    if evidence.get("preprocessingSha256") != _sha256(preprocessing_path):
        raise ValueError("iOS preprocessing provenance does not match the repository")

    return {
        "status": "validated_physical_iphone_coreml"
        if not device["simulator"]
        else "validated_simulator_coreml",
        "captured_at_utc": evidence["capturedAtUTC"],
        "app_version": evidence["appVersion"],
        "device": device,
        "backend": "CoreML",
        "requested_compute_units": "all",
        "latency": latency,
        "output_max_abs_drift": evidence["outputMaxAbsDrift"],
        "thermal_state_before": evidence["thermalStateBefore"],
        "thermal_state_after": evidence["thermalStateAfter"],
        "power_measurement": "not_measured",
        "neural_engine_placement": "not_measured",
        "claim_boundary": (
            "Physical iPhone Core ML application latency; not proof of Apple Neural Engine "
            "placement and not a power measurement."
        ),
    }


def write_report(summary: dict[str, Any], output_json: Path, output_markdown: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latency = summary["latency"]
    device = summary["device"]
    lines = [
        "# EdgeGenBench physical iPhone Core ML report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Device: `{device['model']}`",
        f"- OS: `{device['systemName']} {device['systemVersion']}`",
        f"- Backend: `{summary['backend']}` (requested compute units: `all`)",
        f"- Cold latency: `{latency['coldMs']:.6f} ms`",
        f"- Warm mean latency: `{latency['warmMeanMs']:.6f} ms`",
        f"- Warm p95 latency: `{latency['warmP95Ms']:.6f} ms`",
        f"- Warm runs: `{latency['warmRuns']}`",
        f"- Output max absolute drift: `{summary['output_max_abs_drift']}`",
        "- Thermal state: "
        f"`{summary['thermal_state_before']}` → `{summary['thermal_state_after']}`",
        "- Power: `not measured`",
        "- Apple Neural Engine placement: `not measured`",
        "",
        f"> {summary['claim_boundary']}",
    ]
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/neural_surrogate/model.pt"))
    parser.add_argument(
        "--preprocessing", type=Path, default=Path("artifacts/neural_surrogate/preprocessing.npz")
    )
    parser.add_argument("--output-json", type=Path, default=Path("reports/ios_device_summary.json"))
    parser.add_argument(
        "--output-markdown", type=Path, default=Path("reports/ios_device_report.md")
    )
    parser.add_argument("--allow-simulator", action="store_true")
    args = parser.parse_args()
    summary = validate_ios_evidence(
        args.evidence,
        model_path=args.model,
        preprocessing_path=args.preprocessing,
        allow_simulator=args.allow_simulator,
    )
    write_report(summary, args.output_json, args.output_markdown)
    print(f"Validated iOS evidence: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

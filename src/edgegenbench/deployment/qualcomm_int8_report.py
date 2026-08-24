"""Build a fail-closed Qualcomm INT8/QDQ evidence report from measured artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from edgegenbench.deployment.qualcomm_ai_hub import (
    assess_qualcomm_int8_candidate,
    calculate_runtime_parity,
    summarize_profile,
)


@dataclass(frozen=True)
class QualcommInt8ReportArtifacts:
    """Paths and acceptance state emitted by the report builder."""

    report_path: Path
    accepted: bool
    rejection_reasons: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_path(manifest_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def build_qualcomm_int8_report(
    manifest_path: Path,
    output_path: Path,
    *,
    max_normalized_drift: float = 0.01,
) -> QualcommInt8ReportArtifacts:
    """Validate a measured manifest and write a recruiter-verifiable report."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")

    source_path = _required_path(manifest_path, manifest.get("source_model"), "source_model")
    quantized_path = _required_path(
        manifest_path, manifest.get("quantized_model"), "quantized_model"
    )
    reference_path = _required_path(
        manifest_path, manifest.get("reference_predictions"), "reference_predictions"
    )
    candidate_path = _required_path(
        manifest_path, manifest.get("candidate_predictions"), "candidate_predictions"
    )
    context_path = _required_path(
        manifest_path, manifest.get("qnn_context_binary"), "qnn_context_binary"
    )

    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object")
    profiles_payload = manifest.get("profiles")
    if not isinstance(profiles_payload, list):
        raise ValueError("profiles must be a list")
    profiles = tuple(
        summarize_profile(item["payload"], int(item["batch_size"]))
        for item in profiles_payload
        if isinstance(item, dict)
    )
    if len(profiles) != len(profiles_payload):
        raise ValueError("each profile must be an object")

    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    parity = calculate_runtime_parity(reference, candidate)
    acceptance = assess_qualcomm_int8_candidate(
        source_model_sha256=_sha256(source_path),
        quantized_model_sha256=_sha256(quantized_path),
        calibration_partition=str(calibration.get("partition", "")),
        calibration_sample_count=int(calibration.get("sample_count", 0)),
        profiles=profiles,
        parity=parity,
        max_normalized_drift=max_normalized_drift,
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": "EdgeGenBench Qualcomm-native INT8/QDQ acceptance",
        "claim_status": "accepted" if acceptance.accepted else "rejected",
        "acceptance": acceptance.to_dict(),
        "artifacts": {
            "source_model": {"path": str(source_path), "sha256": _sha256(source_path)},
            "quantized_model": {
                "path": str(quantized_path),
                "sha256": _sha256(quantized_path),
            },
            "qnn_context_binary": {
                "path": str(context_path),
                "sha256": _sha256(context_path),
                "size_bytes": context_path.stat().st_size,
            },
        },
        "profiles": [profile.to_dict() for profile in profiles],
        "parity": parity.to_dict(),
        "limitations": [
            "Profile latency is not end-to-end Android latency.",
            "This report makes no power claim.",
            "Acceptance requires exclusive NPU placement and all required batches.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return QualcommInt8ReportArtifacts(
        report_path=output_path,
        accepted=acceptance.accepted,
        rejection_reasons=acceptance.rejection_reasons,
    )

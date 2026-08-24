from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edgegenbench.deployment.qualcomm_int8_report import (
    build_qualcomm_int8_report,
    validate_tracked_qualcomm_int8_report,
)


def _manifest(tmp_path: Path, *, cpu_node: bool = False) -> Path:
    (tmp_path / "source.onnx").write_bytes(b"source")
    (tmp_path / "quantized.onnx").write_bytes(b"quantized")
    (tmp_path / "context.bin").write_bytes(b"qnn-context")
    reference = np.arange(18, dtype=np.float32).reshape(3, 6)
    np.save(tmp_path / "reference.npy", reference)
    np.save(tmp_path / "candidate.npy", reference + np.float32(1e-4))
    units = [{"compute_unit": "NPU"}]
    if cpu_node:
        units.append({"compute_unit": "CPU"})
    manifest = {
        "source_model": "source.onnx",
        "quantized_model": "quantized.onnx",
        "qnn_context_binary": "context.bin",
        "reference_predictions": "reference.npy",
        "candidate_predictions": "candidate.npy",
        "calibration": {"partition": "train", "sample_count": 4200},
        "profiles": [
            {
                "batch_size": batch,
                "payload": {
                    "execution_summary": {
                        "estimated_inference_time": float(batch + 10),
                        "estimated_inference_peak_memory": 1000 + batch,
                    },
                    "nodes": units,
                },
            }
            for batch in (1, 32, 256)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_accepts_complete_exclusive_npu_evidence(tmp_path: Path) -> None:
    artifacts = build_qualcomm_int8_report(_manifest(tmp_path), tmp_path / "report.json")
    assert artifacts.accepted
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report["claim_status"] == "accepted"
    assert report["artifacts"]["qnn_context_binary"]["sha256"]
    assert [item["batch_size"] for item in report["profiles"]] == [1, 32, 256]


def test_rejects_any_cpu_placement(tmp_path: Path) -> None:
    artifacts = build_qualcomm_int8_report(
        _manifest(tmp_path, cpu_node=True), tmp_path / "report.json"
    )
    assert not artifacts.accepted
    assert any("exclusive NPU" in reason for reason in artifacts.rejection_reasons)


def test_tracked_device_rejection_matches_frozen_gate() -> None:
    accepted, reasons = validate_tracked_qualcomm_int8_report(
        Path("reports/qualcomm_int8_qnn_v0_1.json")
    )
    assert not accepted
    assert reasons == ("held-out normalized drift exceeds the preregistered limit",)

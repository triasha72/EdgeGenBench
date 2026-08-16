"""Build the tracked Qualcomm QNN deployment evidence report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_MODEL = Path("artifacts/neural_onnx/neural_surrogate.onnx")
ARTIFACT_ROOT = Path("artifacts/qualcomm_ai_hub")
OUTPUT_PATH = Path("reports/qualcomm_qnn_v0_1.json")


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return value


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _baseline_batch(
    batch_size: int,
) -> dict[str, Any]:
    source = _load_json(ARTIFACT_ROOT / "batch_provenance" / f"batch{batch_size}.json")

    return {
        "batch_size": batch_size,
        "compile_job_id": source["compile_job_id"],
        "profile_job_id": source["profile_job_id"],
        "target_model_id": source["target_model_id"],
        "target_model_type": source["target_model_type"],
        "serialized_model_size_bytes": source["serialized_model_size_bytes"],
        "serialized_model_sha256": source["serialized_model_sha256"],
        "compute_units": source["compute_units"],
        "estimated_inference_time_us": source["estimated_inference_time_us"],
        "estimated_inference_peak_memory_bytes": source["estimated_inference_peak_memory_bytes"],
        "target_metadata": source["target_metadata"],
    }


def main() -> None:
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Missing source model: {SOURCE_MODEL}")

    heldout_baseline = _load_json(ARTIFACT_ROOT / "heldout_parity_batch1" / "summary.json")
    multigraph = _load_json(ARTIFACT_ROOT / "multigraph" / "multigraph.json")
    multigraph_validation = _load_json(ARTIFACT_ROOT / "multigraph" / "validation.json")

    local_r2 = float(heldout_baseline["local_mean_r2"])
    remote_r2 = float(heldout_baseline["remote_mean_r2"])
    local_nrmse = float(heldout_baseline["local_mean_nrmse_std"])
    remote_nrmse = float(heldout_baseline["remote_mean_nrmse_std"])

    report = {
        "schema_version": "0.2",
        "experiment": ("EdgeGenBench Qualcomm QNN Snapdragon 8 Elite deployment"),
        "source_model": {
            "path": str(SOURCE_MODEL),
            "sha256": _sha256(SOURCE_MODEL),
            "input_name": "features",
            "input_width": 10,
            "output_width": 6,
            "source_precision": "float32",
        },
        "hardware": {
            "device": "Snapdragon 8 Elite QRD",
            "device_os": "15",
            "chipset": "qualcomm-snapdragon-8-elite",
            "chipset_alias": "sm8750",
            "soc_model": "69",
            "backend": "HTP",
            "hexagon": "v79",
            "qairt_version": "2.45.0.260326154327",
        },
        "batch_specific_baseline": {
            "1": _baseline_batch(1),
            "32": _baseline_batch(32),
            "256": _baseline_batch(256),
        },
        "heldout_batch1_baseline": {
            **heldout_baseline,
            "r2_delta": remote_r2 - local_r2,
            "nrmse_delta": remote_nrmse - local_nrmse,
        },
        "linked_multigraph": {
            "compile_jobs": multigraph["compile_jobs"],
            "link_job_id": multigraph["link_job_id"],
            "link_status": multigraph["link_status"],
            "target_model_id": multigraph["target_model_id"],
            "target_model_type": multigraph["target_model_type"],
            "target_metadata": multigraph["target_metadata"],
            "validation": multigraph_validation,
        },
        "measurement_semantics": {
            "latency": ("Qualcomm AI Hub profile estimated model inference time"),
            "throughput": (
                "Derived from configured batch size and AI Hub estimated inference time"
            ),
            "runtime_memory": ("Qualcomm AI Hub estimated inference peak memory"),
            "serialized_model_size": ("Downloaded QNN Context Binary byte size"),
            "allclose": (
                "Diagnostic only; predictive quality and normalized drift are reported separately"
            ),
        },
        "claim_boundaries": [
            ("Results apply to the named Snapdragon 8 Elite QRD configuration."),
            ("AI Hub profile latency is not end-to-end Android application latency."),
            (
                "Derived throughput is based on model-profile latency and "
                "is not an end-to-end system throughput measurement."
            ),
            ("CPU, CoreML, and Snapdragon timings are not treated as same-hardware comparisons."),
            ("The QNN baseline uses float32 external I/O with HTP FP16-relaxed execution."),
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

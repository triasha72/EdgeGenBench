#!/usr/bin/env python3
"""Validate tracked deployment evidence and build the final portfolio acceptance matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
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


def _repository_sha256(repository_root: Path, source_path: Path) -> tuple[str | None, str]:
    """Hash the committed artifact, falling back to the worktree outside Git."""
    try:
        relative_path = source_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError("source model must be inside the repository") from exc

    result = subprocess.run(
        ["git", "-C", str(repository_root), "show", f"HEAD:{relative_path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        return hashlib.sha256(result.stdout).hexdigest(), "committed_git_blob"
    if source_path.is_file():
        return _sha256(source_path), "worktree_file"
    return None, "missing"


def validate_ai_hub_qnn(report_path: Path, repository_root: Path) -> dict[str, Any]:
    report = _load_json(report_path)
    if report.get("schema_version") != "0.2":
        raise ValueError("unsupported Qualcomm AI Hub report")
    source = report.get("source_model")
    hardware = report.get("hardware")
    linked = report.get("linked_multigraph")
    if (
        not isinstance(source, dict)
        or not isinstance(hardware, dict)
        or not isinstance(linked, dict)
    ):
        raise ValueError("incomplete Qualcomm AI Hub identity")
    source_path = repository_root / str(source.get("path"))
    current_source_sha256, source_hash_origin = _repository_sha256(repository_root, source_path)
    source_model_matches = current_source_sha256 == source.get("sha256")
    if hardware.get("backend") != "HTP" or not hardware.get("qairt_version"):
        raise ValueError("Qualcomm report must identify the HTP backend and QAIRT version")
    if "SUCCESS" not in str(linked.get("link_status")):
        raise ValueError("linked QNN context job did not succeed")
    context_path = repository_root / str(linked.get("serialized_model_path"))
    context_sha256, context_hash_origin = _repository_sha256(repository_root, context_path)
    context_size = context_path.stat().st_size if context_path.is_file() else None
    context_matches = context_sha256 == linked.get(
        "serialized_model_sha256"
    ) and context_size == linked.get("serialized_model_size_bytes")
    if not context_matches:
        raise ValueError("tracked QNN context binary does not match its reported hash and size")
    validation = linked.get("validation")
    if not isinstance(validation, dict) or validation.get("device") != hardware.get("device"):
        raise ValueError("linked QNN validation device does not match the report")
    graphs = validation.get("graphs")
    if not isinstance(graphs, dict) or set(graphs) != {
        "edgegenbench_batch1",
        "edgegenbench_batch32",
        "edgegenbench_batch256",
    }:
        raise ValueError("linked QNN evidence must contain batch 1, 32, and 256 graphs")

    graph_summaries = []
    for name, graph in graphs.items():
        if not isinstance(graph, dict):
            raise ValueError(f"invalid QNN graph evidence: {name}")
        profile = graph.get("profile")
        parity = graph.get("parity")
        if not isinstance(profile, dict) or profile.get("compute_units") != {"NPU": 9}:
            raise ValueError(f"{name} does not prove exclusive NPU compute-unit mapping")
        if not isinstance(parity, dict) or float(parity.get("max_normalized_drift", 1.0)) > 0.012:
            raise ValueError(f"{name} exceeds the predeclared normalized drift limit")
        if not graph.get("profile_job_id") or not graph.get("inference_job_id"):
            raise ValueError(f"{name} requires profile and inference job IDs")
        graph_summaries.append(
            {
                "graph": name,
                "batch_size": graph["batch_size"],
                "latency_ms": profile["estimated_inference_latency_ms"],
                "throughput_samples_per_second": profile["estimated_throughput_samples_per_second"],
                "peak_memory_bytes": profile["estimated_inference_peak_memory_bytes"],
                "compute_units": profile["compute_units"],
                "max_normalized_drift": parity["max_normalized_drift"],
                "profile_job_id": graph["profile_job_id"],
                "inference_job_id": graph["inference_job_id"],
            }
        )
    return {
        "status": (
            "validated_ai_hub_physical_qnn"
            if source_model_matches
            else "tracked_ai_hub_report_model_provenance_mismatch"
        ),
        "device": hardware["device"],
        "backend": "QNN HTP",
        "qairt_version": hardware["qairt_version"],
        "context_model_id": linked["target_model_id"],
        "context_path": linked["serialized_model_path"],
        "context_sha256": context_sha256,
        "context_size_bytes": context_size,
        "context_hash_origin": context_hash_origin,
        "context_matches_repository": context_matches,
        "link_job_id": linked["link_job_id"],
        "graphs": sorted(graph_summaries, key=lambda value: value["batch_size"]),
        "reported_source_model_sha256": source.get("sha256"),
        "current_source_model_sha256": current_source_sha256,
        "source_model_hash_origin": source_hash_origin,
        "source_model_matches_repository": source_model_matches,
        "claim_boundary": "Physical AI Hub model profiling; not Android APK end-to-end latency.",
    }


def validate_android_16kb_runtime(evidence_dir: Path, report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise ValueError("tracked Android 16 KB report is missing")
    device_report = evidence_dir / "device-report.txt"
    measurement_status = evidence_dir / "measurement-status.txt"
    csv_path = evidence_dir / "latency-memory.csv"
    if not all(path.is_file() for path in (device_report, measurement_status, csv_path)):
        raise ValueError("Android 16 KB evidence bundle is incomplete")

    device = dict(
        line.split("=", 1)
        for line in device_report.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    if device.get("page_size") != "16384" or device.get("abi") != "arm64-v8a":
        raise ValueError("Android runtime evidence must prove an ARM64 16 KB environment")
    if device.get("model") != "sdk_gphone16k_arm64" or device.get("sdk") != "35":
        raise ValueError("Android runtime evidence must identify the API 35 16 KB emulator")

    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise ValueError("Android 16 KB evidence must contain exactly ten runs")
    for row in rows:
        if float(row["preprocess_max_abs_drift"]) > 1e-6:
            raise ValueError("Android 16 KB preprocessing drift exceeds tolerance")
        if float(row["output_max_abs_drift"]) > 1e-6:
            raise ValueError("Android 16 KB output drift exceeds tolerance")

    status = measurement_status.read_text(encoding="utf-8")
    if "backend=reference" not in status or "power=not-measured" not in status:
        raise ValueError("Android 16 KB claim boundaries are missing")
    logs = sorted((evidence_dir / "runs").glob("run-*/logcat.txt"))
    if len(logs) != 10:
        raise ValueError("Android 16 KB evidence must retain ten logcat files")
    for log in logs:
        content = log.read_text(encoding="utf-8")
        if '"runtime_page_size_bytes":16384' not in content:
            raise ValueError(f"{log.name} does not report a 16 KB runtime")
        if "EdgeGenBench: backend=reference" not in content or "CPU fallback=false" not in content:
            raise ValueError(f"{log.name} has invalid backend placement metadata")

    return {
        "status": "validated_16kb_emulator_runtime",
        "environment": "Android 15 API 35 ARM64 16 KB emulator",
        "page_size_bytes": 16384,
        "runs": len(rows),
        "report": f"reports/{report_path.name}",
        "claim": (
            "APK/JNI reference path executed on PAGE_SIZE=16384; not physical-device performance."
        ),
    }


def build_portfolio_acceptance(
    *, repository_root: Path, qnn_report: Path, output_json: Path, output_markdown: Path
) -> dict[str, Any]:
    qnn = validate_ai_hub_qnn(qnn_report, repository_root)
    android_report = repository_root / "reports/android_sm_a356e_reference_10_run_v0_1_4.md"
    if not android_report.is_file():
        raise ValueError("tracked Samsung reference report is missing")
    android_16kb_report = repository_root / "reports/android_16kb_emulator_reference_v0_1_7.md"
    android_16kb = validate_android_16kb_runtime(
        repository_root / "reports/device/android-16kb-api35-reference-10-runs",
        android_16kb_report,
    )
    matrix = {
        "schema_version": 1,
        "project": "EdgeGenBench",
        "lanes": {
            "native_cpp": {
                "status": "validated_in_ci",
                "claim": "C++17 reference runtime, tests, CLI, and fused preprocessing acceptance.",
            },
            "android_reference": {
                "status": "validated_physical_device",
                "device": "Samsung SM-A356E",
                "report": android_report.relative_to(repository_root).as_posix(),
                "claim": "Reference JNI/application measurements; not QNN.",
            },
            "qualcomm_ai_hub_qnn": qnn,
            "android_qnn_apk": {
                "status": "implementation_complete_evidence_pending",
                "claim": "Build/JNI/capture paths exist; requires a supported Snapdragon APK run.",
            },
            "android_16kb_runtime": android_16kb,
            "power": {
                "status": "not_measured",
                "claim": "No power-savings claim is made without a named calibrated tool.",
            },
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EdgeGenBench portfolio acceptance",
        "",
        "| Evidence lane | Status | Claim boundary |",
        "|---|---|---|",
    ]
    for name, lane in matrix["lanes"].items():
        claim = lane.get("claim", lane.get("claim_boundary"))
        lines.append(f"| `{name}` | `{lane['status']}` | {claim} |")
    lines.extend(
        [
            "",
            "## Validated Qualcomm QNN results",
            "",
            f"Device: **{qnn['device']}**; backend: **{qnn['backend']}**; "
            f"QAIRT: `{qnn['qairt_version']}`.",
            f"Source-model provenance match: **{qnn['source_model_matches_repository']}**.",
            f"Tracked QNN context provenance match: **{qnn['context_matches_repository']}** "
            f"(`{qnn['context_path']}`, `{qnn['context_sha256']}`).",
            "",
            "| Batch | AI Hub latency (ms) | Throughput (samples/s) | "
            "Peak memory (bytes) | Placement | Max normalized drift |",
            "|---:|---:|---:|---:|---|---:|",
        ]
    )
    for graph in qnn["graphs"]:
        lines.append(
            f"| {graph['batch_size']} | {graph['latency_ms']:.6f} | "
            f"{graph['throughput_samples_per_second']:.3f} | {graph['peak_memory_bytes']} | "
            f"NPU × 9 | {graph['max_normalized_drift']:.9f} |"
        )
    lines.extend(
        [
            "",
            "AI Hub measurements are physical-device model profiles, not Android "
            "application end-to-end timings. Current-model acceptance requires source-model "
            "provenance to match the repository, as reported above.",
            "Power remains unmeasured. The remaining hardware proof item is a supported-device "
            "QNN APK run; the 16 KB reference APK/JNI runtime is validated on an API 35 emulator.",
        ]
    )
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--qnn-report", type=Path, default=Path("reports/qualcomm_qnn_v0_1.json"))
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/portfolio_acceptance.json")
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=Path("reports/portfolio_acceptance.md")
    )
    args = parser.parse_args()
    build_portfolio_acceptance(
        repository_root=args.repository_root.resolve(),
        qnn_report=args.qnn_report,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(f"Portfolio acceptance written to {args.output_markdown}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate tracked deployment evidence and build the final portfolio acceptance matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    current_source_sha256 = _sha256(source_path) if source_path.is_file() else None
    source_model_matches = current_source_sha256 == source.get("sha256")
    if hardware.get("backend") != "HTP" or not hardware.get("qairt_version"):
        raise ValueError("Qualcomm report must identify the HTP backend and QAIRT version")
    if "SUCCESS" not in str(linked.get("link_status")):
        raise ValueError("linked QNN context job did not succeed")
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
        "link_job_id": linked["link_job_id"],
        "graphs": sorted(graph_summaries, key=lambda value: value["batch_size"]),
        "reported_source_model_sha256": source.get("sha256"),
        "current_source_model_sha256": current_source_sha256,
        "source_model_matches_repository": source_model_matches,
        "claim_boundary": "Physical AI Hub model profiling; not Android APK end-to-end latency.",
    }


def build_portfolio_acceptance(
    *, repository_root: Path, qnn_report: Path, output_json: Path, output_markdown: Path
) -> dict[str, Any]:
    qnn = validate_ai_hub_qnn(qnn_report, repository_root)
    android_report = repository_root / "reports/android_sm_a356e_reference_10_run_v0_1_4.md"
    if not android_report.is_file():
        raise ValueError("tracked Samsung reference report is missing")
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
            "android_16kb_runtime": {
                "status": "packaging_validated_runtime_pending",
                "claim": "ELF/APK alignment passes; runtime PAGE_SIZE=16384 evidence is pending.",
            },
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
            "Power remains unmeasured. The two pending proof items are a supported-device "
            "QNN APK run and a runtime page size of 16384 bytes.",
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

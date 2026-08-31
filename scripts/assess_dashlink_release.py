#!/usr/bin/env python3
"""Generate the deployment acceptance decision for the real DASHlink model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(artifact: dict[str, object]) -> dict[str, object]:
    report = artifact["evaluations"]["test"]["classification_report"]
    onnx = artifact["onnx"]
    minority_recall = min(float(report[str(index)]["recall"]) for index in (1, 2, 3))
    checks = {
        "minimum_macro_f1": {"value": report["macro avg"]["f1-score"], "minimum": 0.75},
        "minimum_anomaly_class_recall": {"value": minority_recall, "minimum": 0.60},
        "minimum_onnx_label_agreement": {"value": onnx["label_agreement"], "minimum": 0.999},
        "maximum_probability_error": {
            "value": onnx["max_probability_absolute_error"],
            "maximum": 1e-5,
        },
        "maximum_cpu_latency_ms_per_row": {
            "value": onnx["cpu_mean_per_row_latency_ms"],
            "maximum": 0.01,
        },
        "maximum_model_size_bytes": {"value": onnx["size_bytes"], "maximum": 40_000_000},
    }
    for check in checks.values():
        threshold = check.get("minimum", check.get("maximum"))
        check["passed"] = (
            check["value"] >= threshold if "minimum" in check else check["value"] <= threshold
        )
    return {
        "schema_version": "1.0",
        "policy": "dashlink-edge-release-v1",
        "decision": "approved" if all(item["passed"] for item in checks.values()) else "rejected",
        "checks": checks,
        "required_runtime_guards": {
            "input_shape": [160, 20],
            "maximum_missing_fraction": 0.05,
            "finite_window_endpoints": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.artifact.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train an edge-friendly baseline on recorded NASA DASHlink approaches."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from edgegenbench.real_data.dashlink import aircraft_groups, load_dashlink, summarize_windows


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def export_and_validate_onnx(model, features, test, output):
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    graph = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, features.shape[1]]))],
        options={id(model): {"zipmap": False}},
        target_opset=18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(graph.SerializeToString())
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    sample = features[test[: min(2_000, len(test))]].astype(np.float32)
    expected_labels = model.predict(sample)
    expected_probabilities = model.predict_proba(sample)
    observed_labels, observed_probabilities = session.run(None, {"features": sample})
    started = time.perf_counter()
    repeats = 20
    for _ in range(repeats):
        session.run(None, {"features": sample})
    elapsed = time.perf_counter() - started
    return {
        "path": output.name,
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "parity_rows": len(sample),
        "label_agreement": float(np.mean(expected_labels == observed_labels)),
        "max_probability_absolute_error": float(
            np.max(np.abs(expected_probabilities - observed_probabilities))
        ),
        "cpu_mean_batch_latency_ms": elapsed * 1_000 / repeats,
        "cpu_mean_per_row_latency_ms": elapsed * 1_000 / (repeats * len(sample)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/dashlink_real_baseline_v1.json")
    )
    parser.add_argument(
        "--onnx-output", type=Path, default=Path("artifacts/dashlink_real_baseline_v1.onnx")
    )
    args = parser.parse_args()
    windows, labels, metadata = load_dashlink(args.data, args.metadata)
    features = summarize_windows(windows)
    groups = aircraft_groups(metadata)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_val, test = next(outer.split(features, labels, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=args.seed)
    train_rel, val_rel = next(
        inner.split(features[train_val], labels[train_val], groups[train_val])
    )
    train = train_val[train_rel]
    validation = train_val[val_rel]
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(features[train], labels[train])
    records = {}
    for name, index in (("validation", validation), ("test", test)):
        prediction = model.predict(features[index])
        records[name] = {
            "rows": len(index),
            "classification_report": classification_report(
                labels[index], prediction, labels=[0, 1, 2, 3], output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                labels[index], prediction, labels=[0, 1, 2, 3]
            ).tolist(),
        }
    payload = {
        "schema_version": "1.0",
        "dataset": "NASA DASHlink Curated 4 Class Anomaly Detection Data Set",
        "source_page": "https://c3.ndc.nasa.gov/dashlink/resources/1018/",
        "model": "Random Forest on 100 window-summary features",
        "seed": args.seed,
        "split_policy": "grouped by de-identified aircraft identifier prefix",
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "source_sha256": {"data": sha256(args.data), "metadata": sha256(args.metadata)},
        "evaluations": records,
        "contains_synthetic_data": False,
    }
    payload["onnx"] = export_and_validate_onnx(model, features, test, args.onnx_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(records["test"]["classification_report"]["macro avg"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

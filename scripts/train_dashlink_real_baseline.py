#!/usr/bin/env python3
"""Train an edge-friendly baseline on recorded NASA DASHlink approaches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from edgegenbench.real_data.dashlink import aircraft_groups, load_dashlink, summarize_windows


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/dashlink_real_baseline_v1.json")
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
    model = HistGradientBoostingClassifier(
        max_iter=150, l2_regularization=1.0, random_state=args.seed, class_weight="balanced"
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
        "model": "HistGradientBoosting on 100 window-summary features",
        "seed": args.seed,
        "split_policy": "grouped by de-identified aircraft identifier prefix",
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "source_sha256": {"data": sha256(args.data), "metadata": sha256(args.metadata)},
        "evaluations": records,
        "contains_synthetic_data": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(records["test"]["classification_report"]["macro avg"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

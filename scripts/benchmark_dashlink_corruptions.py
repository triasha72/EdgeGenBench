#!/usr/bin/env python3
"""Measure the real DASHlink ONNX model under controlled sensor corruption."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from edgegenbench.real_data.dashlink import aircraft_groups, load_dashlink, summarize_windows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_test_indices(features, labels, groups, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    _, test = next(splitter.split(features, labels, groups))
    return test


def evaluate(session, windows, labels):
    features = summarize_windows(windows)
    prediction = session.run(None, {"features": features})[0]
    return prediction, {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    windows, labels, metadata = load_dashlink(args.data, args.metadata)
    groups = aircraft_groups(metadata)
    index = grouped_test_indices(np.empty((len(windows), 1)), labels, groups, args.seed)
    real_windows = np.asarray(windows[index], dtype=np.float32)
    test_labels = labels[index]
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    clean_prediction, clean_metrics = evaluate(session, real_windows, test_labels)
    generator = np.random.default_rng(args.seed)
    channel_scale = np.nanstd(real_windows, axis=(0, 1), keepdims=True)

    noisy = real_windows + generator.normal(size=real_windows.shape).astype(np.float32) * (
        channel_scale * 0.01
    )
    noisy_prediction, noisy_metrics = evaluate(session, noisy, test_labels)
    del noisy

    missing = real_windows.copy()
    missing_mask = generator.random(missing.shape) < 0.02
    missing_mask[:, (0, -1), :] = False
    missing[missing_mask] = np.nan
    missing_prediction, missing_metrics = evaluate(session, missing, test_labels)
    del missing

    temporal = real_windows.copy()
    temporal[:, 78:82, :] = np.nan
    temporal_prediction, temporal_metrics = evaluate(session, temporal, test_labels)
    del temporal

    conditions = {
        "clean": {**clean_metrics, "prediction_consistency_with_clean": 1.0},
        "gaussian_noise_1pct_channel_std": {
            **noisy_metrics,
            "prediction_consistency_with_clean": float(
                np.mean(noisy_prediction == clean_prediction)
            ),
        },
        "random_missing_2pct": {
            **missing_metrics,
            "prediction_consistency_with_clean": float(
                np.mean(missing_prediction == clean_prediction)
            ),
        },
        "four_timestep_dropout": {
            **temporal_metrics,
            "prediction_consistency_with_clean": float(
                np.mean(temporal_prediction == clean_prediction)
            ),
        },
    }
    result = {
        "schema_version": "1.0",
        "evidence_label": "dashlink_real_sensor_corruption_benchmark",
        "dataset_rows": len(test_labels),
        "dataset_sha256": sha256(args.data),
        "metadata_sha256": sha256(args.metadata),
        "model_sha256": sha256(args.model),
        "seed": args.seed,
        "conditions": conditions,
        "contains_synthetic_flights": False,
        "interpretation": "Perturbations are applied only to held-out real recorded flights.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(conditions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

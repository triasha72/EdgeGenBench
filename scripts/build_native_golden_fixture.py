#!/usr/bin/env python3
"""Generate cross-runtime golden vectors from the real EdgeGenBench ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/neural_onnx/neural_surrogate.onnx")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("artifacts/neural_onnx/metadata.json")
    )
    parser.add_argument(
        "--preprocessing", type=Path, default=Path("artifacts/neural_surrogate/preprocessing.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/native_model_golden.json")
    )
    args = parser.parse_args()

    designs = [
        {
            "passenger_capacity": 40.0,
            "design_range_km": 450.0,
            "cruise_speed_kmh": 410.0,
            "battery_specific_energy_wh_per_kg": 300.0,
            "hydrogen_storage_efficiency": 0.45,
            "hybridization_ratio": 0.0,
            "propulsion_architecture": "conventional_turboprop",
        },
        {
            "passenger_capacity": 65.0,
            "design_range_km": 950.0,
            "cruise_speed_kmh": 535.0,
            "battery_specific_energy_wh_per_kg": 525.0,
            "hydrogen_storage_efficiency": 0.575,
            "hybridization_ratio": 0.25,
            "propulsion_architecture": "parallel_hybrid",
        },
        {
            "passenger_capacity": 82.0,
            "design_range_km": 1400.0,
            "cruise_speed_kmh": 570.0,
            "battery_specific_energy_wh_per_kg": 650.0,
            "hydrogen_storage_efficiency": 0.68,
            "hybridization_ratio": 0.70,
            "propulsion_architecture": "series_hybrid",
        },
        {
            "passenger_capacity": 100.0,
            "design_range_km": 2000.0,
            "cruise_speed_kmh": 620.0,
            "battery_specific_energy_wh_per_kg": 750.0,
            "hydrogen_storage_efficiency": 0.78,
            "hybridization_ratio": 1.0,
            "propulsion_architecture": "fuel_cell_electric",
        },
    ]
    with np.load(args.preprocessing, allow_pickle=False) as preprocessing:
        categories = [str(value) for value in preprocessing["categories"].tolist()]
        feature_mean = preprocessing["feature_mean"].astype(np.float32)
        feature_scale = preprocessing["feature_scale"].astype(np.float32)
        target_mean = preprocessing["target_mean"].astype(np.float32)
        target_scale = preprocessing["target_scale"].astype(np.float32)
    numeric_names = (
        "passenger_capacity",
        "design_range_km",
        "cruise_speed_kmh",
        "battery_specific_energy_wh_per_kg",
        "hydrogen_storage_efficiency",
        "hybridization_ratio",
    )
    numeric = np.asarray(
        [[design[name] for name in numeric_names] for design in designs], dtype=np.float32
    )
    one_hot = np.zeros((len(designs), len(categories)), dtype=np.float32)
    for row, design in enumerate(designs):
        one_hot[row, categories.index(str(design["propulsion_architecture"]))] = 1.0
    encoded = np.concatenate(((numeric - feature_mean) / feature_scale, one_hot), axis=1)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    normalized = session.run(["predictions"], {"features": encoded})[0].astype(np.float32)
    physical = normalized * target_scale + target_mean
    payload = {
        "schema_version": "1.0",
        "model_sha256": _sha256(args.model),
        "preprocessing_sha256": _sha256(args.preprocessing),
        "input_name": "features",
        "output_name": "predictions",
        "input_width": 10,
        "output_width": 6,
        "tolerances": {"encoded_atol": 1e-6, "normalized_atol": 1e-5, "physical_atol": 1e-3},
        "cases": [
            {
                "raw": design,
                "encoded": encoded[index].tolist(),
                "normalized_prediction": normalized[index].tolist(),
                "physical_prediction": physical[index].tolist(),
            }
            for index, design in enumerate(designs)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE={args.output}")
    print(f"MODEL_SHA256={payload['model_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

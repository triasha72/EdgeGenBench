#!/usr/bin/env python3
"""Compile, link, profile, and validate the current ONNX model on Qualcomm AI Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import qai_hub as hub

from edgegenbench.deployment.qualcomm_ai_hub import (
    calculate_runtime_parity,
    qnn_graph_option,
    stringify_metadata,
    summarize_profile,
)

NUMERIC_FEATURES = [
    "passenger_capacity",
    "design_range_km",
    "cruise_speed_kmh",
    "battery_specific_energy_wh_per_kg",
    "hydrogen_storage_efficiency",
    "hybridization_ratio",
]
CATEGORICAL_FEATURE = "propulsion_architecture"

MODEL_PATH = Path("artifacts/neural_onnx/neural_surrogate.onnx")
PREPROCESSOR_PATH = Path("artifacts/neural_surrogate/preprocessing.npz")
HELDOUT_PATH = Path("artifacts/neural_surrogate/test_predictions.csv")
OUTPUT_PATH = Path("reports/qualcomm_qnn_v0_1.json")
CONTEXT_PATH = Path("artifacts/qualcomm_ai_hub/current_model/edgegenbench_multigraph.bin")
CHECKPOINT_PATH = Path("artifacts/qualcomm_ai_hub/current_model/rerun_checkpoint.json")

DEVICE_NAME = "Snapdragon 8 Elite QRD"
DEVICE_OS = "15"
GRAPH_BATCHES = {
    "edgegenbench_batch1": 1,
    "edgegenbench_batch32": 32,
    "edgegenbench_batch256": 256,
}
COMPILE_OPTIONS = "--qnn_options default_graph_htp_precision=FLOAT16"
LINK_OPTIONS = "--qnn_options default_graph_htp_optimizations=O=3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _require_success(status: object, label: str) -> None:
    if not bool(getattr(status, "success", False)):
        raise RuntimeError(f"{label} failed: {status}")


def _partition(features: np.ndarray, batch_size: int) -> list[np.ndarray]:
    if len(features) % batch_size:
        raise ValueError(f"Feature count {len(features)} is not divisible by {batch_size}.")
    return [
        features[start : start + batch_size].astype(np.float32, copy=False)
        for start in range(0, len(features), batch_size)
    ]


def _download_output(inference_job: Any) -> tuple[str, np.ndarray]:
    output = inference_job.download_output_data()
    if not isinstance(output, dict) or len(output) != 1:
        raise RuntimeError(f"Expected one AI Hub output tensor, received {type(output)}.")
    name = next(iter(output))
    arrays = []
    for value in output[name]:
        array = np.asarray(value, dtype=np.float32)
        arrays.append(array.reshape(-1, array.shape[-1]))
    return name, np.concatenate(arrays, axis=0)


def _load_preprocessing() -> dict[str, Any]:
    with np.load(PREPROCESSOR_PATH, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _transform_features(frame: pd.DataFrame, preprocessing: dict[str, Any]) -> np.ndarray:
    numeric = frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    standardized = (numeric - preprocessing["feature_mean"]) / preprocessing["feature_scale"]
    categories = [str(value) for value in preprocessing["categories"].tolist()]
    category_to_index = {category: index for index, category in enumerate(categories)}
    indices = np.asarray(
        [category_to_index[str(value)] for value in frame[CATEGORICAL_FEATURE]], dtype=np.int64
    )
    one_hot = np.eye(len(categories), dtype=np.float32)[indices]
    return np.concatenate([standardized, one_hot], axis=1).astype(np.float32)


def _quality(
    actual: np.ndarray, prediction: np.ndarray, preprocessing: dict[str, Any]
) -> dict[str, float]:
    physical = (prediction * preprocessing["target_scale"] + preprocessing["target_mean"]).astype(
        np.float64
    )
    actual64 = actual.astype(np.float64)
    residual = actual64 - physical
    squared_error = np.sum(residual * residual, axis=0)
    centered = actual64 - np.mean(actual64, axis=0)
    total_squared = np.sum(centered * centered, axis=0)
    r2 = 1.0 - squared_error / total_squared
    rmse = np.sqrt(np.mean(residual * residual, axis=0))
    actual_std = np.std(actual64, axis=0)
    nrmse_std = rmse / actual_std
    return {
        "mean_r2": float(np.nanmean(r2)),
        "mean_nrmse_std": float(np.nanmean(nrmse_std)),
    }


def _checkpoint(client: hub.Client, model_sha256: str) -> dict[str, Any]:
    if not CHECKPOINT_PATH.is_file():
        return {"source_model_sha256": model_sha256, "graphs": {}}
    value = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if value.get("source_model_sha256") != model_sha256:
        raise RuntimeError("Checkpoint belongs to a different source model; remove it to rerun.")
    # Validate that the saved linked model remains accessible before resuming.
    if value.get("target_model_id"):
        client.get_model(str(value["target_model_id"]))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse successful saved job IDs.")
    args = parser.parse_args()

    for path in (MODEL_PATH, PREPROCESSOR_PATH, HELDOUT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact is missing: {path}")

    model_sha256 = _sha256(MODEL_PATH)
    client = hub.Client()
    device = hub.Device(DEVICE_NAME, DEVICE_OS)
    checkpoint = (
        _checkpoint(client, model_sha256)
        if args.resume
        else {
            "source_model_sha256": model_sha256,
            "graphs": {},
        }
    )

    if checkpoint.get("target_model_id"):
        target_model = client.get_model(str(checkpoint["target_model_id"]))
        compile_jobs = [client.get_job(job_id) for job_id in checkpoint["compile_job_ids"]]
        link_job = client.get_job(str(checkpoint["link_job_id"]))
        print(f"Resuming linked model {target_model.model_id}", flush=True)
    else:
        graph_names = list(GRAPH_BATCHES)
        print("Submitting three QNN DLC compiles and the dependent link job...", flush=True)
        compile_jobs, link_job = client.submit_compile_and_link_jobs(
            models=[MODEL_PATH] * len(graph_names),
            device=device,
            name="EdgeGenBench current-model QNN multigraph",
            input_specs=[
                {"features": ((batch_size, 10), "float32")} for batch_size in GRAPH_BATCHES.values()
            ],
            graph_names=graph_names,
            compile_options=COMPILE_OPTIONS,
            link_options=LINK_OPTIONS,
        )
        if link_job is None:
            raise RuntimeError("AI Hub did not create the QNN link job.")
        checkpoint.update(
            compile_job_ids=[job.job_id for job in compile_jobs],
            link_job_id=link_job.job_id,
        )
        _write_json(CHECKPOINT_PATH, checkpoint)
        print(f"Compile jobs: {checkpoint['compile_job_ids']}", flush=True)
        print(f"Link job: {link_job.job_id}", flush=True)
        status = link_job.wait()
        _require_success(status, "QNN context link")
        target_model = link_job.get_target_model()
        if target_model is None:
            raise RuntimeError("Successful link job did not return a target model.")
        checkpoint["target_model_id"] = target_model.model_id
        checkpoint["link_status"] = str(status)
        _write_json(CHECKPOINT_PATH, checkpoint)

    CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_model.download(str(CONTEXT_PATH))

    frame = pd.read_csv(HELDOUT_PATH)
    if len(frame) != 900:
        raise RuntimeError(f"Expected 900 held-out rows, received {len(frame)}.")
    preprocessing = _load_preprocessing()
    targets = [str(value) for value in preprocessing["targets"].tolist()]
    features = _transform_features(frame, preprocessing)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    local = np.asarray(session.run([output_name], {input_name: features})[0], dtype=np.float32)

    graph_records: dict[str, Any] = {}
    for graph_name, batch_size in GRAPH_BATCHES.items():
        saved = checkpoint["graphs"].get(graph_name, {}) if args.resume else {}
        if saved.get("profile_job_id"):
            profile_job = client.get_job(str(saved["profile_job_id"]))
        else:
            print(f"Submitting profile for {graph_name}...", flush=True)
            profile_job = client.submit_profile_job(
                model=target_model,
                device=device,
                name=f"EdgeGenBench current-model {graph_name} profile",
                options=qnn_graph_option(graph_name),
            )
            if isinstance(profile_job, list):
                raise RuntimeError("Expected one profile job.")
            saved["profile_job_id"] = profile_job.job_id
            checkpoint["graphs"][graph_name] = saved
            _write_json(CHECKPOINT_PATH, checkpoint)
        profile_status = profile_job.wait()
        _require_success(profile_status, f"Profile {graph_name}")
        profile = summarize_profile(profile_job.download_profile(), batch_size=batch_size)

        sample_count = 900 if batch_size == 1 else 256
        if saved.get("inference_job_id"):
            inference_job = client.get_job(str(saved["inference_job_id"]))
        else:
            print(f"Submitting inference for {graph_name}...", flush=True)
            inference_job = client.submit_inference_job(
                model=target_model,
                device=device,
                inputs={input_name: _partition(features[:sample_count], batch_size)},
                name=f"EdgeGenBench current-model {graph_name} inference",
                options=qnn_graph_option(graph_name),
            )
            if isinstance(inference_job, list):
                raise RuntimeError("Expected one inference job.")
            saved["inference_job_id"] = inference_job.job_id
            _write_json(CHECKPOINT_PATH, checkpoint)
        inference_status = inference_job.wait()
        _require_success(inference_status, f"Inference {graph_name}")
        remote_output_name, remote = _download_output(inference_job)
        reference = local[:sample_count]
        parity = calculate_runtime_parity(reference, remote)
        actual = frame.loc[: sample_count - 1, targets].to_numpy(dtype=np.float32)
        local_quality = _quality(actual, reference, preprocessing)
        remote_quality = _quality(actual, remote, preprocessing)
        graph_records[graph_name] = {
            "graph_name": graph_name,
            "batch_size": batch_size,
            "sample_count": sample_count,
            "profile_job_id": profile_job.job_id,
            "inference_job_id": inference_job.job_id,
            "profile": profile.to_dict(),
            "remote_output_name": remote_output_name,
            "parity": parity.to_dict(),
            "predictive_quality": {
                "local_mean_r2": local_quality["mean_r2"],
                "remote_mean_r2": remote_quality["mean_r2"],
                "r2_delta": remote_quality["mean_r2"] - local_quality["mean_r2"],
                "local_mean_nrmse_std": local_quality["mean_nrmse_std"],
                "remote_mean_nrmse_std": remote_quality["mean_nrmse_std"],
                "nrmse_delta": (remote_quality["mean_nrmse_std"] - local_quality["mean_nrmse_std"]),
            },
        }
        print(f"Completed {graph_name}: {profile.to_dict()}", flush=True)

    metadata = stringify_metadata(target_model.metadata)
    qairt_version = next(
        (value for key, value in metadata.items() if "QAIRT_SDK_VERSION" in key), "unknown"
    )
    backend = next((value for key, value in metadata.items() if key.endswith("BACKEND")), "HTP")
    hexagon = next(
        (value for key, value in metadata.items() if "HEXAGON_VERSION" in key), "unknown"
    )
    report = {
        "schema_version": "0.2",
        "experiment": "EdgeGenBench current-model Qualcomm QNN deployment",
        "source_model": {
            "path": str(MODEL_PATH),
            "sha256": model_sha256,
            "input_name": input_name,
            "input_width": 10,
            "output_name": output_name,
            "output_width": 6,
            "source_precision": "float32",
        },
        "hardware": {
            "device": DEVICE_NAME,
            "device_os": DEVICE_OS,
            "chipset": "qualcomm-snapdragon-8-elite",
            "chipset_alias": "sm8750",
            "backend": backend,
            "hexagon": hexagon,
            "qairt_version": qairt_version,
        },
        "linked_multigraph": {
            "compile_jobs": [
                {
                    "batch": batch,
                    "graph_name": graph,
                    "compile_job_id": job.job_id,
                    "compile_status": str(job.get_status()),
                }
                for (graph, batch), job in zip(GRAPH_BATCHES.items(), compile_jobs, strict=True)
            ],
            "link_job_id": link_job.job_id,
            "link_status": str(link_job.get_status()),
            "target_model_id": target_model.model_id,
            "target_model_type": str(target_model.model_type),
            "serialized_model_size_bytes": CONTEXT_PATH.stat().st_size,
            "serialized_model_sha256": _sha256(CONTEXT_PATH),
            "target_metadata": metadata,
            "validation": {
                "link_job_id": link_job.job_id,
                "target_model_id": target_model.model_id,
                "target_model_type": str(target_model.model_type),
                "target_metadata": metadata,
                "target_input_spec": str(target_model.input_spec),
                "device": DEVICE_NAME,
                "device_os": DEVICE_OS,
                "local_reference": {
                    "model_path": str(MODEL_PATH),
                    "input_name": input_name,
                    "output_name": output_name,
                    "heldout_rows": len(frame),
                },
                "graphs": graph_records,
            },
        },
        "claim_boundary": (
            "Physical Qualcomm AI Hub QNN HTP model measurements; not Android APK "
            "end-to-end latency and not a calibrated power measurement."
        ),
    }
    _write_json(OUTPUT_PATH, report)
    print(f"Wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

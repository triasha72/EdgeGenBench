"""Validate the linked EdgeGenBench QNN model on Snapdragon hardware."""

from __future__ import annotations

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
from edgegenbench.evaluation.regression import calculate_regression_metrics
from edgegenbench.models.fp32_linear import DEFAULT_TARGETS
from edgegenbench.models.neural_preprocessing import NeuralPreprocessor

MODEL_PATH = Path("artifacts/neural_onnx/neural_surrogate.onnx")
PREPROCESSOR_PATH = Path("artifacts/neural_surrogate/preprocessing.npz")
HELDOUT_PATH = Path("artifacts/neural_surrogate/test_predictions.csv")
MULTIGRAPH_METADATA_PATH = Path("artifacts/qualcomm_ai_hub/multigraph/multigraph.json")
OUTPUT_PATH = Path("artifacts/qualcomm_ai_hub/multigraph/validation.json")

DEVICE_NAME = "Snapdragon 8 Elite QRD"
DEVICE_OS = "15"

GRAPH_BATCHES = {
    "edgegenbench_batch1": 1,
    "edgegenbench_batch32": 32,
    "edgegenbench_batch256": 256,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def _require_job_success(status: object, label: str) -> None:
    if not bool(getattr(status, "success", False)):
        raise RuntimeError(f"{label} did not complete successfully: {status}")


def _partition_features(
    features: np.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features must have two dimensions.")

    if len(features) % batch_size != 0:
        raise ValueError(f"Selected feature count must be divisible by batch size {batch_size}.")

    return [
        features[start : start + batch_size].astype(
            np.float32,
            copy=False,
        )
        for start in range(0, len(features), batch_size)
    ]


def _download_output(
    inference_job: Any,
) -> tuple[str, np.ndarray]:
    output_data = inference_job.download_output_data()

    if not isinstance(output_data, dict):
        raise RuntimeError(f"Unexpected AI Hub output type: {type(output_data)}")

    if len(output_data) != 1:
        raise RuntimeError(f"Expected exactly one model output, received {list(output_data)}.")

    output_name = next(iter(output_data))
    values = output_data[output_name]
    arrays: list[np.ndarray] = []

    for value in values:
        array = np.asarray(value, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(1, -1)
        else:
            array = array.reshape(-1, array.shape[-1])

        arrays.append(array)

    if not arrays:
        raise RuntimeError("AI Hub inference returned no arrays.")

    return output_name, np.concatenate(arrays, axis=0)


def _mean_regression_metrics(
    actual: pd.DataFrame,
    prediction: np.ndarray,
    preprocessor: NeuralPreprocessor,
) -> dict[str, float]:
    physical = preprocessor.inverse_transform_targets(prediction)

    prediction_frame = pd.DataFrame(
        physical,
        columns=DEFAULT_TARGETS,
    )

    metrics = calculate_regression_metrics(
        actual=actual,
        predicted=prediction_frame,
        targets=DEFAULT_TARGETS,
    )

    return {
        "mean_r2": float(np.nanmean(metrics["r2"])),
        "mean_nrmse_std": float(np.nanmean(metrics["nrmse_std"])),
    }


def main() -> None:
    metadata = _read_json(MULTIGRAPH_METADATA_PATH)
    target_model_id = str(metadata["target_model_id"])
    link_job_id = str(metadata["link_job_id"])

    client = hub.Client()
    target_model = client.get_model(target_model_id)
    device = hub.Device(DEVICE_NAME, DEVICE_OS)

    available_graphs = {
        str(graph_name) for graph_name in target_model.input_spec if graph_name is not None
    }
    expected_graphs = set(GRAPH_BATCHES)

    if available_graphs != expected_graphs:
        raise RuntimeError(
            "Linked QNN graph mismatch: "
            f"expected={sorted(expected_graphs)}, "
            f"actual={sorted(available_graphs)}"
        )

    frame = pd.read_csv(HELDOUT_PATH)

    if len(frame) != 900:
        raise RuntimeError(f"Expected 900 held-out rows, received {len(frame)}.")

    preprocessor = NeuralPreprocessor.load(PREPROCESSOR_PATH)
    features = preprocessor.transform_features(frame)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    local_full = session.run(
        [output_name],
        {input_name: features.astype(np.float32)},
    )[0]
    local_full = np.asarray(local_full, dtype=np.float32)

    validation: dict[str, Any] = {
        "link_job_id": link_job_id,
        "target_model_id": target_model_id,
        "target_model_type": str(target_model.model_type),
        "target_metadata": stringify_metadata(target_model.metadata),
        "target_input_spec": str(target_model.input_spec),
        "device": DEVICE_NAME,
        "device_os": DEVICE_OS,
        "local_reference": {
            "model_path": str(MODEL_PATH),
            "input_name": input_name,
            "output_name": output_name,
            "heldout_rows": int(len(frame)),
        },
        "graphs": {},
    }

    for graph_name, batch_size in GRAPH_BATCHES.items():
        print()
        print("=" * 72)
        print(f"GRAPH={graph_name} BATCH={batch_size}")
        print("=" * 72)

        options = qnn_graph_option(graph_name)

        profile_job = client.submit_profile_job(
            model=target_model,
            device=device,
            name=f"EdgeGenBench linked QNN {graph_name} profile",
            options=options,
        )

        if isinstance(profile_job, list):
            raise RuntimeError("Expected one profile job.")

        print(f"PROFILE_JOB_ID={profile_job.job_id}")
        profile_status = profile_job.wait()
        print(f"PROFILE_STATUS={profile_status}")
        _require_job_success(profile_status, f"Profile for {graph_name}")

        profile_payload = profile_job.download_profile()
        profile_summary = summarize_profile(
            profile_payload,
            batch_size=batch_size,
        )

        sample_count = 900 if batch_size == 1 else 256
        selected_features = features[:sample_count]
        selected_local = local_full[:sample_count]
        input_entries = _partition_features(selected_features, batch_size)

        inference_job = client.submit_inference_job(
            model=target_model,
            device=device,
            inputs={input_name: input_entries},
            name=f"EdgeGenBench linked QNN {graph_name} inference",
            options=options,
        )

        if isinstance(inference_job, list):
            raise RuntimeError("Expected one inference job.")

        print(f"INFERENCE_JOB_ID={inference_job.job_id}")
        inference_status = inference_job.wait()
        print(f"INFERENCE_STATUS={inference_status}")
        _require_job_success(inference_status, f"Inference for {graph_name}")

        remote_output_name, remote = _download_output(inference_job)

        if remote.shape != selected_local.shape:
            raise RuntimeError(
                "Linked-QNN output-shape mismatch: "
                f"local={selected_local.shape}, remote={remote.shape}."
            )

        parity = calculate_runtime_parity(selected_local, remote)

        actual = frame.loc[
            : sample_count - 1,
            list(DEFAULT_TARGETS),
        ].reset_index(drop=True)

        local_quality = _mean_regression_metrics(
            actual,
            selected_local,
            preprocessor,
        )
        remote_quality = _mean_regression_metrics(
            actual,
            remote,
            preprocessor,
        )

        graph_record = {
            "graph_name": graph_name,
            "batch_size": batch_size,
            "sample_count": sample_count,
            "profile_job_id": profile_job.job_id,
            "inference_job_id": inference_job.job_id,
            "profile": profile_summary.to_dict(),
            "remote_output_name": remote_output_name,
            "parity": parity.to_dict(),
            "predictive_quality": {
                "local_mean_r2": local_quality["mean_r2"],
                "remote_mean_r2": remote_quality["mean_r2"],
                "r2_delta": (remote_quality["mean_r2"] - local_quality["mean_r2"]),
                "local_mean_nrmse_std": local_quality["mean_nrmse_std"],
                "remote_mean_nrmse_std": remote_quality["mean_nrmse_std"],
                "nrmse_delta": (remote_quality["mean_nrmse_std"] - local_quality["mean_nrmse_std"]),
            },
        }

        validation["graphs"][graph_name] = graph_record

        print()
        print(json.dumps(graph_record, indent=2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

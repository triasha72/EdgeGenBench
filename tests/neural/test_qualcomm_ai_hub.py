"""Offline tests for Qualcomm AI Hub/QNN deployment helpers."""

from __future__ import annotations

import numpy as np
import pytest

from edgegenbench.deployment.qualcomm_ai_hub import (
    assess_qualcomm_int8_candidate,
    calculate_runtime_parity,
    collect_compute_units,
    qnn_graph_option,
    stringify_metadata,
    summarize_profile,
)


def test_qnn_graph_option() -> None:
    assert (
        qnn_graph_option("edgegenbench_batch32")
        == "--qnn_options context_enable_graphs=edgegenbench_batch32"
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "bad graph",
        "bad;graph",
        "bad\ngraph",
    ],
)
def test_qnn_graph_option_rejects_invalid_names(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        qnn_graph_option(value)


def test_stringify_metadata() -> None:
    metadata = {
        1: "HTP",
        "hexagon": "v79",
    }

    assert stringify_metadata(metadata) == {
        "1": "HTP",
        "hexagon": "v79",
    }


def test_collect_compute_units() -> None:
    profile = {
        "execution_detail": [
            {"name": "Input", "compute_unit": "NPU"},
            {"name": "Linear_0", "compute_unit": "NPU"},
            {"name": "Linear_1", "compute_unit": "NPU"},
            {"name": "Output", "compute_unit": "CPU"},
        ]
    }

    assert collect_compute_units(profile) == {
        "CPU": 1,
        "NPU": 3,
    }


def test_summarize_profile() -> None:
    profile = {
        "execution_summary": {
            "estimated_inference_time": 35,
            "estimated_inference_peak_memory": 122978304,
        },
        "execution_detail": [{"compute_unit": "NPU"} for _ in range(9)],
    }

    summary = summarize_profile(profile, batch_size=32)

    assert summary.estimated_inference_time_us == 35.0
    assert summary.estimated_inference_latency_ms == pytest.approx(0.035)
    assert summary.estimated_throughput_samples_per_second == pytest.approx(32_000_000.0 / 35.0)
    assert summary.estimated_inference_peak_memory_bytes == 122978304
    assert summary.compute_units == {"NPU": 9}


def test_summarize_profile_rejects_bad_batch() -> None:
    with pytest.raises(ValueError):
        summarize_profile({}, batch_size=0)


def test_runtime_parity_exact_match() -> None:
    reference = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    metrics = calculate_runtime_parity(reference, reference.copy())

    assert metrics.sample_count == 2
    assert metrics.output_width == 2
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.max_abs_error == 0.0
    assert metrics.mean_normalized_drift == 0.0
    assert metrics.max_normalized_drift == 0.0
    assert metrics.allclose_rtol_1e3_atol_1e3 is True


def test_runtime_parity_small_drift() -> None:
    reference = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )

    candidate = reference + np.float32(1.0e-4)

    metrics = calculate_runtime_parity(reference, candidate)

    assert metrics.mae == pytest.approx(1.0e-4, rel=1.0e-3)
    assert metrics.max_abs_error < 2.0e-4
    assert metrics.allclose_rtol_1e3_atol_1e3 is True


def test_runtime_parity_rejects_shape_mismatch() -> None:
    reference = np.zeros((2, 6), dtype=np.float32)
    candidate = np.zeros((1, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="Prediction-shape mismatch"):
        calculate_runtime_parity(reference, candidate)


def test_qualcomm_int8_gate_accepts_complete_measured_evidence() -> None:
    parity = calculate_runtime_parity(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[1.001, 2.001], [3.001, 4.001]], dtype=np.float32),
    )
    profiles = tuple(
        summarize_profile(
            {
                "execution_summary": {
                    "estimated_inference_time": latency,
                    "estimated_inference_peak_memory": 120_000_000,
                },
                "execution_detail": [{"compute_unit": "NPU"} for _ in range(9)],
            },
            batch_size=batch,
        )
        for batch, latency in ((1, 30), (32, 35), (256, 60))
    )

    result = assess_qualcomm_int8_candidate(
        source_model_sha256="a" * 64,
        quantized_model_sha256="b" * 64,
        calibration_partition="train",
        calibration_sample_count=4200,
        profiles=profiles,
        parity=parity,
    )

    assert result.accepted is True
    assert result.rejection_reasons == ()
    assert result.profiled_batch_sizes == (1, 32, 256)


def test_qualcomm_int8_gate_rejects_leakage_and_incomplete_placement() -> None:
    parity = calculate_runtime_parity(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([[0.0], [1.0]], dtype=np.float32),
    )
    profile = summarize_profile(
        {
            "execution_summary": {
                "estimated_inference_time": 30,
                "estimated_inference_peak_memory": 120_000_000,
            },
            "execution_detail": [
                {"compute_unit": "NPU"},
                {"compute_unit": "CPU"},
            ],
        },
        batch_size=1,
    )

    result = assess_qualcomm_int8_candidate(
        source_model_sha256="a" * 64,
        quantized_model_sha256="b" * 64,
        calibration_partition="test",
        calibration_sample_count=900,
        profiles=(profile,),
        parity=parity,
    )

    assert result.accepted is False
    assert any("training partition" in reason for reason in result.rejection_reasons)
    assert any("required batch" in reason for reason in result.rejection_reasons)
    assert any("exclusive NPU" in reason for reason in result.rejection_reasons)

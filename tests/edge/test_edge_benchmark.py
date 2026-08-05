"""Tests for edge equivalence and latency benchmarking."""

from typing import Any

import pandas as pd
import pytest

from edgegenbench.deployment.benchmark import (
    benchmark_edge_models,
)


def test_edge_benchmark_creates_outputs(
    edge_bundle: dict[str, Any],
) -> None:
    """The benchmark should save equivalence and latency reports."""
    export = edge_bundle["export"]

    artifacts = benchmark_edge_models(
        dataset_path=edge_bundle["dataset_path"],
        surrogate_model_path=edge_bundle["surrogate_path"],
        feasibility_model_path=edge_bundle["classifier_path"],
        surrogate_onnx_path=(export.surrogate_onnx_path),
        feasibility_onnx_path=(export.feasibility_onnx_path),
        metadata_path=(export.metadata_path),
        output_dir=(edge_bundle["root"] / "benchmark"),
        batch_sizes=(1, 8, 16),
        repeats=2,
        warmups=1,
    )

    assert artifacts.equivalence_path.exists()
    assert artifacts.latency_path.exists()
    assert artifacts.summary_path.exists()

    equivalence = pd.read_csv(artifacts.equivalence_path)

    latency = pd.read_csv(artifacts.latency_path)

    assert len(equivalence) == 4
    assert not latency.empty

    assert artifacts.classifier_agreement == pytest.approx(1.0)

    assert artifacts.max_surrogate_absolute_error < 1.0e-2

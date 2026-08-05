"""Tests for ONNX model export."""

import json
from typing import Any

import onnx


def test_onnx_export_creates_valid_models(
    edge_bundle: dict[str, Any],
) -> None:
    """Both exported ONNX graphs should be valid."""
    artifacts = edge_bundle["export"]

    assert artifacts.surrogate_onnx_path.exists()
    assert artifacts.feasibility_onnx_path.exists()
    assert artifacts.metadata_path.exists()

    onnx.checker.check_model(onnx.load(artifacts.surrogate_onnx_path))

    onnx.checker.check_model(onnx.load(artifacts.feasibility_onnx_path))

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["feature_encoder"]["feature_count"] == artifacts.feature_count

    assert len(metadata["surrogate"]["targets"]) == 3

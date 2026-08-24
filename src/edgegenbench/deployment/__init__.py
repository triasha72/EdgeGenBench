"""Edge deployment tools with lazy optional-runtime imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EdgeBenchmarkArtifacts",
    "EdgeExportArtifacts",
    "EdgeFeatureEncoder",
    "OnnxFeasibilityClassifier",
    "OnnxSurrogate",
    "benchmark_edge_models",
    "export_edge_models",
]

_EXPORTS = {
    "EdgeBenchmarkArtifacts": ("edgegenbench.deployment.benchmark", "EdgeBenchmarkArtifacts"),
    "benchmark_edge_models": ("edgegenbench.deployment.benchmark", "benchmark_edge_models"),
    "EdgeFeatureEncoder": ("edgegenbench.deployment.feature_encoder", "EdgeFeatureEncoder"),
    "EdgeExportArtifacts": ("edgegenbench.deployment.onnx_export", "EdgeExportArtifacts"),
    "export_edge_models": ("edgegenbench.deployment.onnx_export", "export_edge_models"),
    "OnnxFeasibilityClassifier": (
        "edgegenbench.deployment.onnx_inference",
        "OnnxFeasibilityClassifier",
    ),
    "OnnxSurrogate": ("edgegenbench.deployment.onnx_inference", "OnnxSurrogate"),
}


def __getattr__(name: str) -> Any:
    """Load one public symbol without importing every optional backend."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

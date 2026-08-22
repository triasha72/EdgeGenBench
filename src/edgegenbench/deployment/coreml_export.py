"""Core ML export and native-app contract for the neural surrogate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from edgegenbench.models.fp32_linear import CATEGORICAL_FEATURE, NUMERIC_FEATURES
from edgegenbench.models.neural_preprocessing import NeuralPreprocessor
from edgegenbench.models.neural_surrogate import load_neural_surrogate_checkpoint

COREML_INPUT_NAME = "features"
COREML_OUTPUT_NAME = "predictions"


@dataclass(frozen=True)
class CoreMLExportArtifacts:
    """Files consumed by the native iOS demo."""

    model_path: Path
    contract_path: Path
    input_dim: int
    output_dim: int


def build_ios_contract(preprocessor: NeuralPreprocessor) -> dict[str, Any]:
    """Serialize preprocessing and inverse-scaling rules for Swift."""
    return {
        "schemaVersion": "1.0",
        "inputName": COREML_INPUT_NAME,
        "outputName": COREML_OUTPUT_NAME,
        "numericFeatures": list(NUMERIC_FEATURES),
        "categoricalFeature": CATEGORICAL_FEATURE,
        "categories": list(preprocessor.categories),
        "featureMean": preprocessor.feature_mean.astype(float).tolist(),
        "featureScale": preprocessor.feature_scale.astype(float).tolist(),
        "targets": list(preprocessor.targets),
        "targetMean": preprocessor.target_mean.astype(float).tolist(),
        "targetScale": preprocessor.target_scale.astype(float).tolist(),
        "inputDimension": preprocessor.input_dim,
        "outputDimension": preprocessor.output_dim,
    }


def export_neural_surrogate_coreml(
    model_path: Path,
    preprocessing_path: Path,
    output_dir: Path = Path("artifacts/coreml"),
) -> CoreMLExportArtifacts:
    """Export a fixed-batch FP16 ML Program and its checked Swift contract."""
    try:
        import coremltools as ct
    except ImportError as exc:  # pragma: no cover - depends on macOS export environment
        raise RuntimeError(
            "Core ML export requires the 'coreml' extra: pip install -e '.[neural,coreml]'"
        ) from exc

    model, targets = load_neural_surrogate_checkpoint(model_path)
    preprocessor = NeuralPreprocessor.load(preprocessing_path)
    if model.config.input_dim != preprocessor.input_dim:
        raise ValueError("Model and preprocessor input dimensions do not match.")
    if model.config.output_dim != preprocessor.output_dim or targets != preprocessor.targets:
        raise ValueError("Model and preprocessor output contracts do not match.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros((1, preprocessor.input_dim), dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=COREML_INPUT_NAME, shape=example.shape)],
        outputs=[ct.TensorType(name=COREML_OUTPUT_NAME)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    model_destination = output_dir / "NeuralSurrogate.mlpackage"
    contract_destination = output_dir / "ModelContract.json"
    converted.save(str(model_destination))
    contract_destination.write_text(
        json.dumps(build_ios_contract(preprocessor), indent=2) + "\n",
        encoding="utf-8",
    )
    return CoreMLExportArtifacts(
        model_path=model_destination,
        contract_path=contract_destination,
        input_dim=preprocessor.input_dim,
        output_dim=preprocessor.output_dim,
    )


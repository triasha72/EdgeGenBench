"""ONNX export for the compact EdgeGenBench neural surrogate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import onnx
import torch

from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)
from edgegenbench.models.neural_surrogate import (
    load_neural_surrogate_checkpoint,
)

DEFAULT_NEURAL_ONNX_OPSET = 18
NEURAL_ONNX_INPUT_NAME = "features"
NEURAL_ONNX_OUTPUT_NAME = "predictions"


@dataclass(frozen=True)
class NeuralOnnxExportArtifacts:
    """Artifacts created by neural-surrogate ONNX export."""

    onnx_path: Path
    metadata_path: Path
    input_dim: int
    output_dim: int
    targets: tuple[str, ...]
    target_opset: int
    onnx_size_bytes: int


def export_neural_surrogate_onnx(
    model_path: Path,
    preprocessing_path: Path,
    output_dir: Path = Path("artifacts/neural_onnx"),
    target_opset: int = DEFAULT_NEURAL_ONNX_OPSET,
) -> NeuralOnnxExportArtifacts:
    """Export a trained PyTorch neural surrogate to ONNX."""
    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"Neural preprocessing artifact does not exist: {preprocessing_path}"
        )

    if target_opset < 1:
        raise ValueError("target_opset must be positive.")

    model, targets = load_neural_surrogate_checkpoint(model_path)

    preprocessor = NeuralPreprocessor.load(preprocessing_path)

    if preprocessor.input_dim != model.config.input_dim:
        raise ValueError("Preprocessor input dimension does not match the neural checkpoint.")

    if preprocessor.output_dim != model.config.output_dim:
        raise ValueError("Preprocessor output dimension does not match the neural checkpoint.")

    if tuple(preprocessor.targets) != targets:
        raise ValueError("Preprocessor targets do not match the neural checkpoint.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    onnx_path = output_dir / "neural_surrogate.onnx"

    metadata_path = output_dir / "metadata.json"

    model.eval()

    example_input = torch.zeros(
        1,
        model.config.input_dim,
        dtype=torch.float32,
    )

    batch_dimension = torch.export.Dim("batch")

    torch.onnx.export(
        model,
        (example_input,),
        onnx_path,
        input_names=[
            NEURAL_ONNX_INPUT_NAME,
        ],
        output_names=[
            NEURAL_ONNX_OUTPUT_NAME,
        ],
        opset_version=target_opset,
        dynamo=True,
        dynamic_shapes={
            "features": {
                0: batch_dimension,
            }
        },
        external_data=False,
    )

    if not onnx_path.exists():
        raise RuntimeError("PyTorch ONNX export did not create the expected model file.")

    onnx_model = onnx.load(str(onnx_path))

    onnx.checker.check_model(onnx_model)

    if len(onnx_model.graph.input) != 1:
        raise RuntimeError("Expected exactly one ONNX input.")

    if len(onnx_model.graph.output) != 1:
        raise RuntimeError("Expected exactly one ONNX output.")

    graph_input = onnx_model.graph.input[0]
    graph_output = onnx_model.graph.output[0]

    if graph_input.name != NEURAL_ONNX_INPUT_NAME:
        raise RuntimeError("Unexpected neural ONNX input name.")

    if graph_output.name != NEURAL_ONNX_OUTPUT_NAME:
        raise RuntimeError("Unexpected neural ONNX output name.")

    input_dimensions = graph_input.type.tensor_type.shape.dim

    output_dimensions = graph_output.type.tensor_type.shape.dim

    if len(input_dimensions) != 2:
        raise RuntimeError("Expected a rank-two ONNX input.")

    if len(output_dimensions) != 2:
        raise RuntimeError("Expected a rank-two ONNX output.")

    if input_dimensions[1].dim_value != model.config.input_dim:
        raise RuntimeError("ONNX input feature dimension does not match the checkpoint.")

    if output_dimensions[1].dim_value != model.config.output_dim:
        raise RuntimeError("ONNX output dimension does not match the checkpoint.")

    if not input_dimensions[0].dim_param:
        raise RuntimeError("ONNX input batch dimension is not dynamic.")

    if not output_dimensions[0].dim_param:
        raise RuntimeError("ONNX output batch dimension is not dynamic.")

    metadata = {
        "schema_version": "1.0.0",
        "generated_at": (datetime.now(UTC).isoformat()),
        "source_framework": "pytorch",
        "deployment_format": "onnx",
        "source_model_path": str(model_path),
        "preprocessing_path": str(preprocessing_path),
        "onnx_model_path": str(onnx_path),
        "target_opset": target_opset,
        "input_name": (NEURAL_ONNX_INPUT_NAME),
        "output_name": (NEURAL_ONNX_OUTPUT_NAME),
        "input_dim": model.config.input_dim,
        "output_dim": model.config.output_dim,
        "hidden_dims": list(model.config.hidden_dims),
        "targets": list(targets),
        "dynamic_batch": True,
        "onnx_size_bytes": int(onnx_path.stat().st_size),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return NeuralOnnxExportArtifacts(
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        input_dim=model.config.input_dim,
        output_dim=model.config.output_dim,
        targets=targets,
        target_opset=target_opset,
        onnx_size_bytes=int(onnx_path.stat().st_size),
    )

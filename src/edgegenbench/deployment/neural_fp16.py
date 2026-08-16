"""FP16 conversion and static-batch specialization for neural ONNX models."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto
from onnxconverter_common import float16

DEFAULT_COREML_BATCH_SIZES = (
    1,
    32,
    256,
)


@dataclass(frozen=True)
class NeuralFp16ExportArtifacts:
    """Artifacts created by FP32-to-FP16 neural ONNX conversion."""

    onnx_path: Path
    metadata_path: Path
    input_dim: int
    output_dim: int
    fp32_model_size_bytes: int
    fp16_model_size_bytes: int
    size_reduction_percent: float
    fp16_initializer_count: int


@dataclass(frozen=True)
class StaticOnnxArtifact:
    """A static-batch ONNX artifact created for provider-specific benchmarking."""

    onnx_path: Path
    batch_size: int
    model_size_bytes: int


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError("Metadata must contain a JSON object.")

    return payload


def _validate_single_rank_two_io(
    model: onnx.ModelProto,
) -> tuple[
    onnx.ValueInfoProto,
    onnx.ValueInfoProto,
]:
    """Validate the single-input/single-output rank-two neural ONNX interface."""
    if len(model.graph.input) != 1:
        raise ValueError("Expected exactly one ONNX graph input.")

    if len(model.graph.output) != 1:
        raise ValueError("Expected exactly one ONNX graph output.")

    graph_input = model.graph.input[0]

    graph_output = model.graph.output[0]

    input_dimensions = graph_input.type.tensor_type.shape.dim

    output_dimensions = graph_output.type.tensor_type.shape.dim

    if len(input_dimensions) != 2:
        raise ValueError("Expected a rank-two ONNX input.")

    if len(output_dimensions) != 2:
        raise ValueError("Expected a rank-two ONNX output.")

    return (
        graph_input,
        graph_output,
    )


def _default_onnx_opset(
    model: onnx.ModelProto,
) -> int | None:
    """Return the default-domain ONNX opset when present."""
    for opset in model.opset_import:
        if opset.domain in {
            "",
            "ai.onnx",
        }:
            return int(opset.version)

    return None


def export_neural_surrogate_fp16(
    fp32_model_path: Path,
    fp32_metadata_path: Path,
    output_dir: Path = Path("artifacts/neural_fp16"),
) -> NeuralFp16ExportArtifacts:
    """Convert the canonical dynamic FP32 neural ONNX graph to FP16."""
    if not fp32_model_path.exists():
        raise FileNotFoundError(f"FP32 neural ONNX model does not exist: {fp32_model_path}")

    fp32_metadata = _load_json_object(fp32_metadata_path)

    fp32_model = onnx.load(str(fp32_model_path))

    onnx.checker.check_model(fp32_model)

    (
        graph_input,
        graph_output,
    ) = _validate_single_rank_two_io(fp32_model)

    input_dimensions = graph_input.type.tensor_type.shape.dim

    output_dimensions = graph_output.type.tensor_type.shape.dim

    if graph_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Expected the FP32 source model input to use FLOAT.")

    if graph_output.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Expected the FP32 source model output to use FLOAT.")

    if not (input_dimensions[0].dim_param):
        raise ValueError("Expected the FP32 source input batch dimension to be dynamic.")

    if not (output_dimensions[0].dim_param):
        raise ValueError("Expected the FP32 source output batch dimension to be dynamic.")

    input_dim = int(input_dimensions[1].dim_value)

    output_dim = int(output_dimensions[1].dim_value)

    if input_dim < 1:
        raise ValueError("FP32 source input width must be positive.")

    if output_dim < 1:
        raise ValueError("FP32 source output width must be positive.")

    if fp32_metadata.get("input_name") != graph_input.name:
        raise ValueError("FP32 metadata input name does not match the ONNX graph.")

    if fp32_metadata.get("output_name") != graph_output.name:
        raise ValueError("FP32 metadata output name does not match the ONNX graph.")

    if (
        int(
            fp32_metadata.get(
                "input_dim",
                -1,
            )
        )
        != input_dim
    ):
        raise ValueError("FP32 metadata input dimension does not match the ONNX graph.")

    if (
        int(
            fp32_metadata.get(
                "output_dim",
                -1,
            )
        )
        != output_dim
    ):
        raise ValueError("FP32 metadata output dimension does not match the ONNX graph.")

    if fp32_metadata.get("dynamic_batch") is not True:
        raise ValueError("FP32 metadata must identify a dynamic-batch ONNX graph.")

    fp16_model = float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=True,
    )

    onnx.checker.check_model(fp16_model)

    (
        fp16_input,
        fp16_output,
    ) = _validate_single_rank_two_io(fp16_model)

    fp16_input_dimensions = fp16_input.type.tensor_type.shape.dim

    fp16_output_dimensions = fp16_output.type.tensor_type.shape.dim

    if fp16_input.name != graph_input.name:
        raise RuntimeError("FP16 conversion changed the external input name.")

    if fp16_output.name != graph_output.name:
        raise RuntimeError("FP16 conversion changed the external output name.")

    if fp16_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError("FP16 conversion did not preserve FP32 external input type.")

    if fp16_output.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError("FP16 conversion did not preserve FP32 external output type.")

    if not (fp16_input_dimensions[0].dim_param):
        raise RuntimeError("FP16 conversion did not preserve dynamic input batching.")

    if not (fp16_output_dimensions[0].dim_param):
        raise RuntimeError("FP16 conversion did not preserve dynamic output batching.")

    if int(fp16_input_dimensions[1].dim_value) != input_dim:
        raise RuntimeError("FP16 input width does not match the FP32 source model.")

    if int(fp16_output_dimensions[1].dim_value) != output_dim:
        raise RuntimeError("FP16 output width does not match the FP32 source model.")

    fp16_initializer_count = sum(
        initializer.data_type == TensorProto.FLOAT16 for initializer in fp16_model.graph.initializer
    )

    if fp16_initializer_count < 1:
        raise RuntimeError("FP16 conversion produced no FLOAT16 initializers.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fp16_path = output_dir / "neural_surrogate_fp16.onnx"

    metadata_path = output_dir / "metadata.json"

    onnx.save(
        fp16_model,
        str(fp16_path),
    )

    onnx.checker.check_model(onnx.load(str(fp16_path)))

    fp32_model_size_bytes = int(fp32_model_path.stat().st_size)

    fp16_model_size_bytes = int(fp16_path.stat().st_size)

    if fp32_model_size_bytes < 1:
        raise RuntimeError("FP32 source model is empty.")

    size_reduction_percent = (1.0 - fp16_model_size_bytes / fp32_model_size_bytes) * 100.0

    metadata = {
        "schema_version": "1.0.0",
        "generated_at": (datetime.now(UTC).isoformat()),
        "source_framework": (
            fp32_metadata.get(
                "source_framework",
                "pytorch",
            )
        ),
        "source_precision": "fp32",
        "deployment_precision": "fp16",
        "deployment_format": "onnx",
        "converter": ("onnxconverter-common"),
        "converter_version": (version("onnxconverter-common")),
        "keep_io_types": True,
        "source_fp32_model_path": str(fp32_model_path),
        "source_fp32_metadata_path": str(fp32_metadata_path),
        "fp16_model_path": str(fp16_path),
        "input_name": (fp16_input.name),
        "output_name": (fp16_output.name),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "targets": list(
            fp32_metadata.get(
                "targets",
                [],
            )
        ),
        "target_opset": (_default_onnx_opset(fp16_model)),
        "dynamic_batch": True,
        "external_input_dtype": ("float32"),
        "external_output_dtype": ("float32"),
        "internal_precision": ("float16"),
        "fp16_initializer_count": (fp16_initializer_count),
        "fp32_model_size_bytes": (fp32_model_size_bytes),
        "fp16_model_size_bytes": (fp16_model_size_bytes),
        "fp16_over_fp32_size_ratio": (fp16_model_size_bytes / fp32_model_size_bytes),
        "size_reduction_percent": (size_reduction_percent),
    }

    preprocessing_path = fp32_metadata.get("preprocessing_path")

    if preprocessing_path is not None:
        metadata["preprocessing_path"] = preprocessing_path

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return NeuralFp16ExportArtifacts(
        onnx_path=fp16_path,
        metadata_path=metadata_path,
        input_dim=input_dim,
        output_dim=output_dim,
        fp32_model_size_bytes=(fp32_model_size_bytes),
        fp16_model_size_bytes=(fp16_model_size_bytes),
        size_reduction_percent=(size_reduction_percent),
        fp16_initializer_count=(fp16_initializer_count),
    )


def specialize_onnx_batch_dimension(
    source_model_path: Path,
    output_path: Path,
    batch_size: int,
) -> StaticOnnxArtifact:
    """Create a static-batch copy of a rank-two ONNX model."""
    if not (source_model_path.exists()):
        raise FileNotFoundError(f"Source ONNX model does not exist: {source_model_path}")

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    model = onnx.load(str(source_model_path))

    onnx.checker.check_model(model)

    (
        graph_input,
        graph_output,
    ) = _validate_single_rank_two_io(model)

    input_batch_dimension = graph_input.type.tensor_type.shape.dim[0]

    output_batch_dimension = graph_output.type.tensor_type.shape.dim[0]

    input_batch_dimension.ClearField("dim_param")

    input_batch_dimension.dim_value = batch_size

    output_batch_dimension.ClearField("dim_param")

    output_batch_dimension.dim_value = batch_size

    model = onnx.shape_inference.infer_shapes(model)

    onnx.checker.check_model(model)

    (
        static_input,
        static_output,
    ) = _validate_single_rank_two_io(model)

    static_input_batch = static_input.type.tensor_type.shape.dim[0]

    static_output_batch = static_output.type.tensor_type.shape.dim[0]

    if int(static_input_batch.dim_value) != batch_size:
        raise RuntimeError("Static ONNX input batch dimension was not specialized correctly.")

    if int(static_output_batch.dim_value) != batch_size:
        raise RuntimeError("Static ONNX output batch dimension was not specialized correctly.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    onnx.save(
        model,
        str(output_path),
    )

    onnx.checker.check_model(onnx.load(str(output_path)))

    return StaticOnnxArtifact(
        onnx_path=output_path,
        batch_size=batch_size,
        model_size_bytes=int(output_path.stat().st_size),
    )


def export_static_batch_variants(
    source_model_path: Path,
    output_dir: Path,
    batch_sizes: Sequence[int] = (DEFAULT_COREML_BATCH_SIZES),
    filename_prefix: str = ("neural_fp16"),
) -> tuple[
    StaticOnnxArtifact,
    ...,
]:
    """Create validated static-batch ONNX variants for runtime benchmarking."""
    normalized_batch_sizes = tuple(
        sorted({int(batch_size) for batch_size in batch_sizes if int(batch_size) > 0})
    )

    if not (normalized_batch_sizes):
        raise ValueError("At least one positive batch size is required.")

    artifacts: list[StaticOnnxArtifact] = []

    for batch_size in normalized_batch_sizes:
        artifact = specialize_onnx_batch_dimension(
            source_model_path=(source_model_path),
            output_path=(output_dir / (f"{filename_prefix}_batch{batch_size}.onnx")),
            batch_size=(batch_size),
        )

        artifacts.append(artifact)

    return tuple(artifacts)

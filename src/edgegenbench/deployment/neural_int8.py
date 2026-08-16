"""Mixed-precision INT8/FP32 QDQ export for the neural ONNX surrogate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import (
    quant_pre_process,
)

from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)

DEFAULT_INT8_OUTPUT_HEAD_NODE = "node_linear_3"
DEFAULT_INT8_CALIBRATION_BATCH_SIZE = 64


@dataclass(frozen=True)
class NeuralInt8ExportArtifacts:
    """Artifacts produced by mixed-precision INT8 neural ONNX export."""

    onnx_path: Path
    metadata_path: Path
    input_dim: int
    output_dim: int
    calibration_rows: int
    fp32_model_size_bytes: int
    int8_model_size_bytes: int
    size_reduction_percent: float
    int8_initializer_count: int
    int32_initializer_count: int
    excluded_nodes: tuple[str, ...]


class _TrainingCalibrationReader(CalibrationDataReader):
    """Feed deterministic training-only calibration batches to ONNX Runtime."""

    def __init__(
        self,
        features: np.ndarray,
        input_name: str,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Calibration batch size must be positive.")

        values = np.asarray(
            features,
            dtype=np.float32,
        )

        if values.ndim != 2:
            raise ValueError("Calibration features must be a rank-two array.")

        if len(values) < 1:
            raise ValueError("Calibration features cannot be empty.")

        if not np.isfinite(values).all():
            raise ValueError("Calibration features contain non-finite values.")

        self._features = values
        self._input_name = input_name
        self._batch_size = int(batch_size)
        self._position: int = 0

        self.rewind()

    def get_next(
        self,
    ) -> dict[str, np.ndarray] | None:
        """Return the next calibration batch."""
        if self._position >= len(self._features):
            return None

        start = self._position

        stop = min(
            start + self._batch_size,
            len(self._features),
        )

        batch = self._features[start:stop]

        self._position = stop

        return {
            self._input_name: batch,
        }

    def rewind(
        self,
    ) -> None:
        """Reset calibration iteration."""
        self._position = 0


def _load_training_frame(
    dataset_path: Path,
) -> pd.DataFrame:
    """Load only the explicit training partition used for INT8 calibration."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    if "split" not in frame.columns:
        raise ValueError("INT8 calibration requires a dataset with a split column.")

    training_frame = frame.loc[frame["split"].astype(str) == "train"].reset_index(drop=True)

    if training_frame.empty:
        raise ValueError("No training rows are available for INT8 calibration.")

    if set(training_frame["split"].astype(str)) != {"train"}:
        raise RuntimeError("INT8 calibration data must contain training rows only.")

    return training_frame


def _validate_dynamic_fp32_interface(
    model: onnx.ModelProto,
    preprocessor: NeuralPreprocessor,
) -> tuple[
    onnx.ValueInfoProto,
    onnx.ValueInfoProto,
]:
    """Validate the canonical dynamic FP32 neural ONNX interface."""
    if len(model.graph.input) != 1:
        raise ValueError("Expected exactly one ONNX graph input.")

    if len(model.graph.output) != 1:
        raise ValueError("Expected exactly one ONNX graph output.")

    graph_input = model.graph.input[0]

    graph_output = model.graph.output[0]

    if graph_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Expected the FP32 source input to use FLOAT.")

    if graph_output.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Expected the FP32 source output to use FLOAT.")

    input_dimensions = graph_input.type.tensor_type.shape.dim

    output_dimensions = graph_output.type.tensor_type.shape.dim

    if len(input_dimensions) != 2:
        raise ValueError("Expected a rank-two ONNX input.")

    if len(output_dimensions) != 2:
        raise ValueError("Expected a rank-two ONNX output.")

    if not (input_dimensions[0].dim_param):
        raise ValueError("Expected a dynamic input batch dimension.")

    if not (output_dimensions[0].dim_param):
        raise ValueError("Expected a dynamic output batch dimension.")

    input_dim = int(input_dimensions[1].dim_value)

    output_dim = int(output_dimensions[1].dim_value)

    if input_dim != preprocessor.input_dim:
        raise ValueError("ONNX input dimension does not match preprocessing.")

    if output_dim != preprocessor.output_dim:
        raise ValueError("ONNX output dimension does not match preprocessing.")

    return (
        graph_input,
        graph_output,
    )


def _validate_mixed_precision_graph(
    model: onnx.ModelProto,
    *,
    output_head_node: str,
) -> tuple[
    int,
    int,
]:
    """Validate INT8 hidden layers with a retained FP32 output head."""
    onnx.checker.check_model(model)

    node_by_name = {node.name: node for node in model.graph.node if node.name}

    if output_head_node not in node_by_name:
        raise RuntimeError(f"Expected FP32 output-head node was not found: {output_head_node}")

    output_head = node_by_name[output_head_node]

    if output_head.op_type != "Gemm":
        raise RuntimeError("Expected the excluded output head to remain a Gemm node.")

    graph_output_names = {output.name for output in model.graph.output}

    if not (set(output_head.output) & graph_output_names):
        raise RuntimeError("Excluded output-head Gemm does not produce the external model output.")

    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}

    for parameter_name in output_head.input[1:]:
        initializer = initializer_by_name.get(parameter_name)

        if initializer is None:
            continue

        if initializer.data_type != TensorProto.FLOAT:
            raise RuntimeError("Excluded output-head parameters must remain FP32.")

    int8_initializer_count = sum(
        initializer.data_type == TensorProto.INT8 for initializer in model.graph.initializer
    )

    int32_initializer_count = sum(
        initializer.data_type == TensorProto.INT32 for initializer in model.graph.initializer
    )

    if int8_initializer_count < 1:
        raise RuntimeError("INT8 export produced no INT8 initializers.")

    if int32_initializer_count < 1:
        raise RuntimeError("INT8 export produced no INT32 bias initializers.")

    quantize_count = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)

    dequantize_count = sum(node.op_type == "DequantizeLinear" for node in model.graph.node)

    if quantize_count < 1 or dequantize_count < 1:
        raise RuntimeError("Expected QDQ operators were not found in the INT8 model.")

    return (
        int8_initializer_count,
        int32_initializer_count,
    )


def export_neural_surrogate_int8(
    fp32_model_path: Path,
    dataset_path: Path,
    preprocessing_path: Path,
    output_dir: Path = Path("artifacts/neural_int8"),
    calibration_batch_size: int = (DEFAULT_INT8_CALIBRATION_BATCH_SIZE),
    output_head_node: str = (DEFAULT_INT8_OUTPUT_HEAD_NODE),
) -> NeuralInt8ExportArtifacts:
    """Export the validated mixed-precision INT8/FP32 neural ONNX model."""
    if not (fp32_model_path.exists()):
        raise FileNotFoundError(f"FP32 neural ONNX model does not exist: {fp32_model_path}")

    if not (preprocessing_path.exists()):
        raise FileNotFoundError(
            f"Neural preprocessing artifact does not exist: {preprocessing_path}"
        )

    if calibration_batch_size < 1:
        raise ValueError("calibration_batch_size must be positive.")

    if not output_head_node:
        raise ValueError("output_head_node cannot be empty.")

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CPUExecutionProvider is required for INT8 calibration.")

    preprocessor = NeuralPreprocessor.load(preprocessing_path)

    source_model = onnx.load(str(fp32_model_path))

    onnx.checker.check_model(source_model)

    (
        graph_input,
        graph_output,
    ) = _validate_dynamic_fp32_interface(
        source_model,
        preprocessor,
    )

    source_nodes = {node.name: node for node in source_model.graph.node if node.name}

    if output_head_node not in source_nodes:
        raise ValueError(f"Configured output-head node does not exist: {output_head_node}")

    if source_nodes[output_head_node].op_type != "Gemm":
        raise ValueError("Configured output-head node must be a Gemm node.")

    training_frame = _load_training_frame(dataset_path)

    calibration_features = preprocessor.transform_features(training_frame).astype(
        np.float32,
        copy=False,
    )

    if calibration_features.shape[1] != preprocessor.input_dim:
        raise RuntimeError("Calibration feature width does not match preprocessing.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    int8_path = output_dir / "neural_surrogate_int8.onnx"

    metadata_path = output_dir / "metadata.json"

    with TemporaryDirectory(prefix=("edgegenbench_int8_")) as temporary_directory:
        preprocessed_path = Path(temporary_directory) / "neural_surrogate_preprocessed.onnx"

        quant_pre_process(
            input_model=(fp32_model_path),
            output_model_path=(preprocessed_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,
        )

        if not (preprocessed_path.exists()):
            raise RuntimeError("ONNX quantization preprocessing did not create a model.")

        preprocessed_model = onnx.load(str(preprocessed_path))

        onnx.checker.check_model(preprocessed_model)

        _validate_dynamic_fp32_interface(
            preprocessed_model,
            preprocessor,
        )

        calibration_reader = _TrainingCalibrationReader(
            features=(calibration_features),
            input_name=(graph_input.name),
            batch_size=(calibration_batch_size),
        )

        quantize_static(
            model_input=(preprocessed_path),
            model_output=(int8_path),
            calibration_data_reader=(calibration_reader),
            quant_format=(QuantFormat.QDQ),
            activation_type=(QuantType.QInt8),
            weight_type=(QuantType.QInt8),
            per_channel=True,
            reduce_range=False,
            calibrate_method=(CalibrationMethod.MinMax),
            nodes_to_exclude=[
                output_head_node,
            ],
            calibration_providers=[
                "CPUExecutionProvider",
            ],
        )

    if not (int8_path.exists()):
        raise RuntimeError("INT8 quantization did not create the expected ONNX model.")

    int8_model = onnx.load(str(int8_path))

    (
        int8_input,
        int8_output,
    ) = _validate_dynamic_fp32_interface(
        int8_model,
        preprocessor,
    )

    if int8_input.name != graph_input.name:
        raise RuntimeError("INT8 conversion changed the external input name.")

    if int8_output.name != graph_output.name:
        raise RuntimeError("INT8 conversion changed the external output name.")

    (
        int8_initializer_count,
        int32_initializer_count,
    ) = _validate_mixed_precision_graph(
        int8_model,
        output_head_node=(output_head_node),
    )

    fp32_model_size_bytes = int(fp32_model_path.stat().st_size)

    int8_model_size_bytes = int(int8_path.stat().st_size)

    if fp32_model_size_bytes < 1:
        raise RuntimeError("FP32 source model is empty.")

    if int8_model_size_bytes < 1:
        raise RuntimeError("INT8 model is empty.")

    size_reduction_percent = (1.0 - (int8_model_size_bytes / fp32_model_size_bytes)) * 100.0

    excluded_nodes = (output_head_node,)

    metadata = {
        "schema_version": ("1.0.0"),
        "generated_at": (datetime.now(UTC).isoformat()),
        "deployment_format": ("onnx"),
        "deployment_precision": ("mixed_int8_fp32"),
        "quantization_format": ("QDQ"),
        "activation_type": ("QInt8"),
        "weight_type": ("QInt8"),
        "weight_quantization": ("per_channel"),
        "per_channel": True,
        "calibration_method": ("MinMax"),
        "calibration_split": ("train"),
        "calibration_rows": int(len(training_frame)),
        "calibration_batch_size": int(calibration_batch_size),
        "excluded_nodes": list(excluded_nodes),
        "output_head_node": (output_head_node),
        "output_head_precision": ("fp32"),
        "mixed_precision": True,
        "reference_provider": ("CPUExecutionProvider"),
        "source_model_path": str(fp32_model_path),
        "dataset_path": str(dataset_path),
        "preprocessing_path": str(preprocessing_path),
        "onnx_model_path": str(int8_path),
        "input_name": (int8_input.name),
        "output_name": (int8_output.name),
        "input_dim": (preprocessor.input_dim),
        "output_dim": (preprocessor.output_dim),
        "targets": list(preprocessor.targets),
        "dynamic_batch": True,
        "external_input_dtype": ("float32"),
        "external_output_dtype": ("float32"),
        "int8_initializer_count": int(int8_initializer_count),
        "int32_initializer_count": int(int32_initializer_count),
        "fp32_model_size_bytes": (fp32_model_size_bytes),
        "int8_model_size_bytes": (int8_model_size_bytes),
        "size_reduction_percent": (size_reduction_percent),
        "onnx_version": version("onnx"),
        "onnxruntime_version": version("onnxruntime"),
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

    return NeuralInt8ExportArtifacts(
        onnx_path=(int8_path),
        metadata_path=(metadata_path),
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
        calibration_rows=int(len(training_frame)),
        fp32_model_size_bytes=(fp32_model_size_bytes),
        int8_model_size_bytes=(int8_model_size_bytes),
        size_reduction_percent=(size_reduction_percent),
        int8_initializer_count=(int8_initializer_count),
        int32_initializer_count=(int32_initializer_count),
        excluded_nodes=(excluded_nodes),
    )

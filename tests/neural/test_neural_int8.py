"""Tests for mixed-precision INT8/FP32 neural ONNX export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
import torch
from onnx import TensorProto

from edgegenbench.deployment.neural_int8 import (
    NeuralInt8ExportArtifacts,
    export_neural_surrogate_int8,
)
from edgegenbench.deployment.neural_onnx_export import (
    NeuralOnnxExportArtifacts,
    export_neural_surrogate_onnx,
)
from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    DEFAULT_TARGETS,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)
from edgegenbench.models.neural_surrogate import (
    NeuralSurrogate,
    NeuralSurrogateConfig,
)


@dataclass(frozen=True)
class Int8Fixture:
    """Representative FP32 and INT8 artifacts for exporter tests."""

    root: Path
    dataset_path: Path
    preprocessing_path: Path
    fp32: NeuralOnnxExportArtifacts
    int8: NeuralInt8ExportArtifacts


def _make_dataset(
    rows: int = 48,
) -> pd.DataFrame:
    """Create a compact dataset with explicit train/test partitions."""
    architectures = (
        "conventional_turboprop",
        "fuel_cell_electric",
        "parallel_hybrid",
        "series_hybrid",
    )

    training_rows = 36

    data: dict[
        str,
        object,
    ] = {
        "passenger_capacity": (
            np.linspace(
                40,
                100,
                rows,
            )
        ),
        "design_range_km": (
            np.linspace(
                500,
                2000,
                rows,
            )
        ),
        "cruise_speed_kmh": (
            np.linspace(
                400,
                650,
                rows,
            )
        ),
        "battery_specific_energy_wh_per_kg": (
            np.linspace(
                300,
                600,
                rows,
            )
        ),
        "hydrogen_storage_efficiency": (
            np.linspace(
                0.5,
                0.8,
                rows,
            )
        ),
        "hybridization_ratio": (
            np.linspace(
                0.0,
                1.0,
                rows,
            )
        ),
        CATEGORICAL_FEATURE: [architectures[index % len(architectures)] for index in range(rows)],
        "split": [("train" if index < training_rows else "test") for index in range(rows)],
    }

    for index, target in enumerate(DEFAULT_TARGETS):
        data[target] = np.linspace(
            100.0 + index,
            300.0 + index,
            rows,
        )

    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def int8_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Int8Fixture:
    """Create representative FP32 and mixed-precision INT8 artifacts."""
    root = tmp_path_factory.mktemp("neural_int8")

    frame = _make_dataset()

    dataset_path = root / "dataset.csv"

    frame.to_csv(
        dataset_path,
        index=False,
    )

    training_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)

    preprocessor = NeuralPreprocessor.fit(training_frame)

    preprocessing_path = root / "preprocessing.npz"

    preprocessor.save(preprocessing_path)

    config = NeuralSurrogateConfig(
        input_dim=(preprocessor.input_dim),
        output_dim=(preprocessor.output_dim),
        hidden_dims=(
            64,
            32,
            16,
        ),
    )

    torch.manual_seed(42)

    model = NeuralSurrogate(config)

    model_path = root / "model.pt"

    torch.save(
        {
            "state_dict": (model.state_dict()),
            "input_dim": (config.input_dim),
            "output_dim": (config.output_dim),
            "hidden_dims": list(config.hidden_dims),
            "targets": list(DEFAULT_TARGETS),
        },
        model_path,
    )

    fp32 = export_neural_surrogate_onnx(
        model_path=(model_path),
        preprocessing_path=(preprocessing_path),
        output_dir=(root / "fp32"),
    )

    int8 = export_neural_surrogate_int8(
        fp32_model_path=(fp32.onnx_path),
        dataset_path=(dataset_path),
        preprocessing_path=(preprocessing_path),
        output_dir=(root / "int8"),
        calibration_batch_size=8,
    )

    return Int8Fixture(
        root=root,
        dataset_path=(dataset_path),
        preprocessing_path=(preprocessing_path),
        fp32=fp32,
        int8=int8,
    )


def test_int8_export_creates_artifacts(
    int8_artifacts: Int8Fixture,
) -> None:
    """INT8 export creates the canonical model and metadata."""
    artifacts = int8_artifacts.int8

    assert artifacts.onnx_path.exists()

    assert artifacts.metadata_path.exists()

    assert artifacts.input_dim == 10

    assert artifacts.output_dim == 6

    assert artifacts.calibration_rows == 36

    assert artifacts.fp32_model_size_bytes > 0

    assert artifacts.int8_model_size_bytes > 0

    assert artifacts.int8_initializer_count > 0

    assert artifacts.int32_initializer_count > 0

    assert artifacts.excluded_nodes == ("node_linear_3",)


def test_int8_graph_passes_onnx_checker(
    int8_artifacts: Int8Fixture,
) -> None:
    """Exported mixed-precision model remains valid ONNX."""
    model = onnx.load(str(int8_artifacts.int8.onnx_path))

    onnx.checker.check_model(model)


def test_int8_preserves_float32_external_io(
    int8_artifacts: Int8Fixture,
) -> None:
    """INT8 hidden layers retain FP32 external I/O."""
    model = onnx.load(str(int8_artifacts.int8.onnx_path))

    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT

    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_int8_preserves_dynamic_batch(
    int8_artifacts: Int8Fixture,
) -> None:
    """Mixed-precision artifact preserves dynamic batching."""
    model = onnx.load(str(int8_artifacts.int8.onnx_path))

    input_dimensions = model.graph.input[0].type.tensor_type.shape.dim

    output_dimensions = model.graph.output[0].type.tensor_type.shape.dim

    assert input_dimensions[0].dim_param

    assert output_dimensions[0].dim_param

    assert input_dimensions[1].dim_value == 10

    assert output_dimensions[1].dim_value == 6


def test_int8_graph_contains_qdq_and_integer_initializers(
    int8_artifacts: Int8Fixture,
) -> None:
    """Hidden-layer QDQ export contains real integer parameters."""
    model = onnx.load(str(int8_artifacts.int8.onnx_path))

    operator_types = {node.op_type for node in model.graph.node}

    assert "QuantizeLinear" in operator_types

    assert "DequantizeLinear" in operator_types

    int8_count = sum(
        initializer.data_type == TensorProto.INT8 for initializer in model.graph.initializer
    )

    int32_count = sum(
        initializer.data_type == TensorProto.INT32 for initializer in model.graph.initializer
    )

    assert int8_count == int8_artifacts.int8.int8_initializer_count

    assert int32_count == int8_artifacts.int8.int32_initializer_count


def test_int8_output_head_remains_fp32(
    int8_artifacts: Int8Fixture,
) -> None:
    """The validation-selected final Gemm remains FP32."""
    model = onnx.load(str(int8_artifacts.int8.onnx_path))

    node_by_name = {node.name: node for node in model.graph.node if node.name}

    assert "node_linear_3" in node_by_name

    output_head = node_by_name["node_linear_3"]

    assert output_head.op_type == "Gemm"

    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}

    assert initializer_by_name["network.6.weight"].data_type == TensorProto.FLOAT

    assert initializer_by_name["network.6.bias"].data_type == TensorProto.FLOAT

    assert "network.6.weight_quantized" not in initializer_by_name


def test_int8_metadata_records_quantization_policy(
    int8_artifacts: Int8Fixture,
) -> None:
    """Metadata records calibration and mixed-precision policy."""
    metadata = json.loads(int8_artifacts.int8.metadata_path.read_text(encoding="utf-8"))

    assert metadata["deployment_precision"] == "mixed_int8_fp32"

    assert metadata["quantization_format"] == "QDQ"

    assert metadata["activation_type"] == "QInt8"

    assert metadata["weight_type"] == "QInt8"

    assert metadata["per_channel"] is True

    assert metadata["calibration_method"] == "MinMax"

    assert metadata["calibration_split"] == "train"

    assert metadata["calibration_rows"] == 36

    assert metadata["excluded_nodes"] == [
        "node_linear_3",
    ]

    assert metadata["output_head_precision"] == "fp32"

    assert metadata["mixed_precision"] is True

    assert metadata["dynamic_batch"] is True


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        4,
        8,
    ],
)
def test_int8_runtime_accepts_dynamic_batches(
    int8_artifacts: Int8Fixture,
    batch_size: int,
) -> None:
    """CPUExecutionProvider executes representative dynamic batches."""
    if "CPUExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CPUExecutionProvider unavailable")

    frame = pd.read_csv(int8_artifacts.dataset_path)

    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)

    preprocessor = NeuralPreprocessor.load(int8_artifacts.preprocessing_path)

    features = preprocessor.transform_features(test_frame).astype(
        np.float32,
        copy=False,
    )

    session = ort.InferenceSession(
        str(int8_artifacts.int8.onnx_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {
            input_name: (features[:batch_size]),
        },
    )[0]

    assert output.shape == (
        batch_size,
        6,
    )

    assert output.dtype == np.float32

    assert np.isfinite(output).all()


def test_int8_export_requires_train_split(
    int8_artifacts: Int8Fixture,
    tmp_path: Path,
) -> None:
    """Calibration cannot silently use non-training data."""
    frame = pd.read_csv(int8_artifacts.dataset_path)

    frame["split"] = "test"

    invalid_dataset = tmp_path / "no_train.csv"

    frame.to_csv(
        invalid_dataset,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=("No training rows are available"),
    ):
        export_neural_surrogate_int8(
            fp32_model_path=(int8_artifacts.fp32.onnx_path),
            dataset_path=(invalid_dataset),
            preprocessing_path=(int8_artifacts.preprocessing_path),
            output_dir=(tmp_path / "invalid"),
        )


def test_int8_export_rejects_missing_dataset(
    int8_artifacts: Int8Fixture,
    tmp_path: Path,
) -> None:
    """Missing calibration dataset produces a clear error."""
    with pytest.raises(
        FileNotFoundError,
        match=("Dataset does not exist"),
    ):
        export_neural_surrogate_int8(
            fp32_model_path=(int8_artifacts.fp32.onnx_path),
            dataset_path=(tmp_path / "missing.csv"),
            preprocessing_path=(int8_artifacts.preprocessing_path),
            output_dir=(tmp_path / "invalid"),
        )


def test_int8_export_rejects_invalid_calibration_batch_size(
    int8_artifacts: Int8Fixture,
    tmp_path: Path,
) -> None:
    """Calibration batch size must be positive."""
    with pytest.raises(
        ValueError,
        match=("calibration_batch_size must be positive"),
    ):
        export_neural_surrogate_int8(
            fp32_model_path=(int8_artifacts.fp32.onnx_path),
            dataset_path=(int8_artifacts.dataset_path),
            preprocessing_path=(int8_artifacts.preprocessing_path),
            output_dir=(tmp_path / "invalid"),
            calibration_batch_size=0,
        )

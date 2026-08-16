"""Tests for FP16 neural ONNX conversion and static-batch specialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto

from edgegenbench.deployment.neural_fp16 import (
    NeuralFp16ExportArtifacts,
    export_neural_surrogate_fp16,
    export_static_batch_variants,
    specialize_onnx_batch_dimension,
)
from edgegenbench.deployment.neural_onnx_export import (
    NeuralOnnxExportArtifacts,
    export_neural_surrogate_onnx,
)
from edgegenbench.models.fp32_linear import (
    NUMERIC_FEATURES,
)
from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)
from edgegenbench.models.neural_surrogate import (
    NeuralSurrogate,
    NeuralSurrogateConfig,
)

TARGETS = (
    "estimated_takeoff_mass_kg",
    "mission_energy_kwh",
    "energy_per_passenger_km_kwh",
    "lifecycle_emissions_proxy_kgco2e",
    "operating_cost_proxy_usd",
    "noise_proxy_db",
)

CATEGORIES = (
    "conventional_turboprop",
    "fuel_cell_electric",
    "parallel_hybrid",
    "series_hybrid",
)


@dataclass(frozen=True)
class Fp16Fixture:
    """Representative FP32 and FP16 artifacts used by this test module."""

    root: Path
    fp32: NeuralOnnxExportArtifacts
    fp16: NeuralFp16ExportArtifacts


@pytest.fixture(scope="module")
def fp16_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Fp16Fixture:
    """Create one representative FP32 and FP16 neural ONNX export."""
    root = tmp_path_factory.mktemp("neural_fp16")

    model_path = root / "model.pt"

    preprocessing_path = root / "preprocessing.npz"

    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
        hidden_dims=(
            64,
            32,
            16,
        ),
    )

    model = NeuralSurrogate(config)

    torch.save(
        {
            "state_dict": (model.state_dict()),
            "input_dim": (config.input_dim),
            "output_dim": (config.output_dim),
            "hidden_dims": list(config.hidden_dims),
            "targets": list(TARGETS),
        },
        model_path,
    )

    numeric_count = len(NUMERIC_FEATURES)

    preprocessor = NeuralPreprocessor(
        categories=(CATEGORIES),
        feature_mean=(
            np.zeros(
                numeric_count,
                dtype=np.float32,
            )
        ),
        feature_scale=(
            np.ones(
                numeric_count,
                dtype=np.float32,
            )
        ),
        target_mean=(
            np.zeros(
                len(TARGETS),
                dtype=np.float32,
            )
        ),
        target_scale=(
            np.ones(
                len(TARGETS),
                dtype=np.float32,
            )
        ),
        targets=(TARGETS),
    )

    preprocessor.save(preprocessing_path)

    fp32_artifacts = export_neural_surrogate_onnx(
        model_path=(model_path),
        preprocessing_path=(preprocessing_path),
        output_dir=(root / "fp32"),
    )

    converted = export_neural_surrogate_fp16(
        fp32_model_path=(fp32_artifacts.onnx_path),
        fp32_metadata_path=(fp32_artifacts.metadata_path),
        output_dir=(root / "fp16"),
    )

    return Fp16Fixture(
        root=root,
        fp32=(fp32_artifacts),
        fp16=(converted),
    )


def test_fp16_conversion_creates_model(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """FP16 conversion creates ONNX and metadata artifacts."""
    artifacts = fp16_artifacts.fp16

    assert artifacts.onnx_path.exists()

    assert artifacts.metadata_path.exists()

    assert artifacts.input_dim == 10

    assert artifacts.output_dim == 6

    assert artifacts.fp32_model_size_bytes > 0

    assert artifacts.fp16_model_size_bytes > 0

    assert artifacts.fp16_initializer_count > 0


def test_fp16_graph_passes_onnx_checker(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """Converted FP16 graph remains a valid ONNX model."""
    model = onnx.load(str(fp16_artifacts.fp16.onnx_path))

    onnx.checker.check_model(model)


def test_fp16_preserves_float32_external_io(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """FP16 internal conversion retains the FP32 deployment boundary."""
    model = onnx.load(str(fp16_artifacts.fp16.onnx_path))

    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT

    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_fp16_contains_float16_initializers(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """Converted model contains FP16 neural parameters."""
    artifacts = fp16_artifacts.fp16

    model = onnx.load(str(artifacts.onnx_path))

    fp16_initializer_count = sum(
        initializer.data_type == TensorProto.FLOAT16 for initializer in model.graph.initializer
    )

    assert fp16_initializer_count > 0

    assert fp16_initializer_count == artifacts.fp16_initializer_count


def test_fp16_preserves_dynamic_batch(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """Canonical FP16 artifact preserves the portable dynamic-batch interface."""
    model = onnx.load(str(fp16_artifacts.fp16.onnx_path))

    input_dimensions = model.graph.input[0].type.tensor_type.shape.dim

    output_dimensions = model.graph.output[0].type.tensor_type.shape.dim

    assert input_dimensions[0].dim_param

    assert output_dimensions[0].dim_param

    assert input_dimensions[1].dim_value == 10

    assert output_dimensions[1].dim_value == 6


def test_fp16_metadata(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """FP16 metadata records conversion provenance and model properties."""
    artifacts = fp16_artifacts.fp16

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["source_precision"] == "fp32"

    assert metadata["deployment_precision"] == "fp16"

    assert metadata["deployment_format"] == "onnx"

    assert metadata["converter"] == "onnxconverter-common"

    assert metadata["keep_io_types"] is True

    assert metadata["dynamic_batch"] is True

    assert metadata["external_input_dtype"] == "float32"

    assert metadata["external_output_dtype"] == "float32"

    assert metadata["internal_precision"] == "float16"

    assert metadata["input_dim"] == 10

    assert metadata["output_dim"] == 6

    assert metadata["targets"] == list(TARGETS)

    assert metadata["fp16_initializer_count"] > 0

    assert metadata["fp32_model_size_bytes"] > 0

    assert metadata["fp16_model_size_bytes"] > 0


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        32,
        256,
    ],
)
def test_static_specialization(
    fp16_artifacts: Fp16Fixture,
    batch_size: int,
) -> None:
    """Static specialization fixes both input and output batch dimensions."""
    static = specialize_onnx_batch_dimension(
        source_model_path=(fp16_artifacts.fp16.onnx_path),
        output_path=(fp16_artifacts.root / "static" / (f"batch{batch_size}.onnx")),
        batch_size=(batch_size),
    )

    model = onnx.load(str(static.onnx_path))

    input_dimensions = model.graph.input[0].type.tensor_type.shape.dim

    output_dimensions = model.graph.output[0].type.tensor_type.shape.dim

    assert static.batch_size == batch_size

    assert static.model_size_bytes > 0

    assert input_dimensions[0].dim_value == batch_size

    assert output_dimensions[0].dim_value == batch_size

    assert not (input_dimensions[0].dim_param)

    assert not (output_dimensions[0].dim_param)

    assert input_dimensions[1].dim_value == 10

    assert output_dimensions[1].dim_value == 6


def test_export_static_batch_variants(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """Convenience exporter creates the canonical CoreML benchmark batch set."""
    variants = export_static_batch_variants(
        source_model_path=(fp16_artifacts.fp16.onnx_path),
        output_dir=(fp16_artifacts.root / "coreml_static"),
    )

    assert [artifact.batch_size for artifact in variants] == [
        1,
        32,
        256,
    ]

    assert all(artifact.onnx_path.exists() for artifact in variants)


def test_static_specialization_rejects_invalid_batch(
    fp16_artifacts: Fp16Fixture,
) -> None:
    """Static specialization rejects non-positive batch sizes."""
    with pytest.raises(
        ValueError,
        match=("batch_size must be positive"),
    ):
        specialize_onnx_batch_dimension(
            source_model_path=(fp16_artifacts.fp16.onnx_path),
            output_path=(fp16_artifacts.root / "invalid.onnx"),
            batch_size=0,
        )

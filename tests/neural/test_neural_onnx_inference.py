"""Tests for neural ONNX Runtime inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from edgegenbench.deployment.neural_onnx_export import (
    export_neural_surrogate_onnx,
)
from edgegenbench.deployment.neural_onnx_inference import (
    NeuralOnnxSurrogate,
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


def _make_frame(
    rows: int = 8,
) -> pd.DataFrame:
    """Create a small representative neural-surrogate dataset."""
    architectures = (
        "conventional_turboprop",
        "parallel_hybrid",
        "series_hybrid",
        "fuel_cell_electric",
    )

    data: dict[str, object] = {
        "passenger_capacity": np.linspace(
            40,
            100,
            rows,
        ),
        "design_range_km": np.linspace(
            500,
            2000,
            rows,
        ),
        "cruise_speed_kmh": np.linspace(
            400,
            600,
            rows,
        ),
        ("battery_specific_energy_wh_per_kg"): np.linspace(
            300,
            600,
            rows,
        ),
        "hydrogen_storage_efficiency": (
            np.linspace(
                0.5,
                0.8,
                rows,
            )
        ),
        "hybridization_ratio": np.linspace(
            0.0,
            1.0,
            rows,
        ),
        CATEGORICAL_FEATURE: [architectures[index % len(architectures)] for index in range(rows)],
    }

    for index, target in enumerate(DEFAULT_TARGETS):
        data[target] = np.linspace(
            10.0 + index,
            20.0 + index,
            rows,
        )

    return pd.DataFrame(data)


def _write_deployment_artifacts(
    directory: Path,
) -> tuple[
    Path,
    Path,
    Path,
]:
    """Create checkpoint, preprocessing, and ONNX test artifacts."""
    torch.manual_seed(42)

    frame = _make_frame()

    preprocessor = NeuralPreprocessor.fit(frame)

    config = NeuralSurrogateConfig(
        input_dim=preprocessor.input_dim,
        output_dim=preprocessor.output_dim,
        hidden_dims=(64, 32, 16),
    )

    model = NeuralSurrogate(config)

    model_path = directory / "model.pt"

    preprocessing_path = directory / "preprocessing.npz"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dims": list(config.hidden_dims),
            "targets": list(DEFAULT_TARGETS),
        },
        model_path,
    )

    preprocessor.save(preprocessing_path)

    export_artifacts = export_neural_surrogate_onnx(
        model_path=model_path,
        preprocessing_path=(preprocessing_path),
        output_dir=(directory / "onnx"),
    )

    return (
        model_path,
        preprocessing_path,
        export_artifacts.onnx_path,
    )


def test_neural_onnx_predict_shape(
    tmp_path: Path,
) -> None:
    (
        _,
        preprocessing_path,
        onnx_path,
    ) = _write_deployment_artifacts(tmp_path)

    metadata_path = tmp_path / "onnx" / "metadata.json"

    runtime = NeuralOnnxSurrogate.load(
        model_path=onnx_path,
        metadata_path=metadata_path,
        preprocessing_path=(preprocessing_path),
    )

    frame = _make_frame()

    predictions = runtime.predict(frame)

    assert predictions.shape == (
        len(frame),
        len(DEFAULT_TARGETS),
    )

    assert tuple(predictions.columns) == tuple(DEFAULT_TARGETS)

    assert predictions.index.equals(frame.index)


def test_neural_onnx_matches_pytorch(
    tmp_path: Path,
) -> None:
    (
        model_path,
        preprocessing_path,
        onnx_path,
    ) = _write_deployment_artifacts(tmp_path)

    metadata_path = tmp_path / "onnx" / "metadata.json"

    runtime = NeuralOnnxSurrogate.load(
        model_path=onnx_path,
        metadata_path=metadata_path,
        preprocessing_path=(preprocessing_path),
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    config = NeuralSurrogateConfig(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dims=tuple(int(value) for value in checkpoint["hidden_dims"]),
    )

    pytorch_model = NeuralSurrogate(config)

    pytorch_model.load_state_dict(checkpoint["state_dict"])

    pytorch_model.eval()

    frame = _make_frame()

    preprocessor = NeuralPreprocessor.load(preprocessing_path)

    features = preprocessor.transform_features(frame)

    with torch.no_grad():
        pytorch_normalized = pytorch_model(torch.from_numpy(features)).numpy().astype(np.float32)

    onnx_normalized = runtime.predict_normalized(frame)

    np.testing.assert_allclose(
        onnx_normalized,
        pytorch_normalized,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    pytorch_physical = preprocessor.inverse_transform_targets(pytorch_normalized)

    onnx_physical = runtime.predict(frame).to_numpy(dtype=np.float32)

    np.testing.assert_allclose(
        onnx_physical,
        pytorch_physical,
        rtol=1.0e-5,
        atol=1.0e-4,
    )

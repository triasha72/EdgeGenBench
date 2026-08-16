"""Tests for neural-surrogate ONNX export."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from edgegenbench.deployment.neural_onnx_export import (
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


def _write_test_artifacts(
    directory: Path,
) -> tuple[Path, Path]:
    model_path = directory / "model.pt"
    preprocessing_path = directory / "preprocessing.npz"

    config = NeuralSurrogateConfig(
        input_dim=10,
        output_dim=6,
        hidden_dims=(64, 32, 16),
    )

    model = NeuralSurrogate(config)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dims": list(config.hidden_dims),
            "targets": list(TARGETS),
        },
        model_path,
    )

    numeric_count = len(NUMERIC_FEATURES)

    preprocessor = NeuralPreprocessor(
        categories=CATEGORIES,
        feature_mean=np.zeros(
            numeric_count,
            dtype=np.float32,
        ),
        feature_scale=np.ones(
            numeric_count,
            dtype=np.float32,
        ),
        target_mean=np.zeros(
            len(TARGETS),
            dtype=np.float32,
        ),
        target_scale=np.ones(
            len(TARGETS),
            dtype=np.float32,
        ),
        targets=TARGETS,
    )

    preprocessor.save(preprocessing_path)

    return (
        model_path,
        preprocessing_path,
    )


def test_export_neural_surrogate_onnx(
    tmp_path: Path,
) -> None:
    model_path, preprocessing_path = _write_test_artifacts(tmp_path)

    artifacts = export_neural_surrogate_onnx(
        model_path=model_path,
        preprocessing_path=(preprocessing_path),
        output_dir=(tmp_path / "onnx"),
    )

    assert artifacts.onnx_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.input_dim == 10
    assert artifacts.output_dim == 6
    assert artifacts.targets == TARGETS
    assert artifacts.onnx_size_bytes > 0

    model = onnx.load(str(artifacts.onnx_path))

    onnx.checker.check_model(model)

    assert model.graph.input[0].name == "features"

    assert model.graph.output[0].name == "predictions"

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["input_dim"] == 10
    assert metadata["output_dim"] == 6

    assert metadata["hidden_dims"] == [
        64,
        32,
        16,
    ]

    assert metadata["targets"] == list(TARGETS)

    assert metadata["dynamic_batch"] is True


def test_exported_model_accepts_dynamic_batches(
    tmp_path: Path,
) -> None:
    model_path, preprocessing_path = _write_test_artifacts(tmp_path)

    artifacts = export_neural_surrogate_onnx(
        model_path=model_path,
        preprocessing_path=(preprocessing_path),
        output_dir=(tmp_path / "onnx"),
    )

    session = ort.InferenceSession(
        str(artifacts.onnx_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    rng = np.random.default_rng(42)

    for batch_size in (
        1,
        4,
        32,
    ):
        features = rng.standard_normal(
            (
                batch_size,
                10,
            )
        ).astype(np.float32)

        predictions = session.run(
            ["predictions"],
            {
                "features": features,
            },
        )[0]

        assert predictions.shape == (
            batch_size,
            6,
        )

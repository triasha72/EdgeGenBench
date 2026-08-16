"""ONNX Runtime inference for the compact neural surrogate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from edgegenbench.models.neural_preprocessing import (
    NeuralPreprocessor,
)


def _load_neural_onnx_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load and validate neural ONNX metadata."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Neural ONNX metadata does not exist: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(metadata, dict):
        raise ValueError("Neural ONNX metadata must contain a JSON object.")

    required_keys = {
        "input_name",
        "output_name",
        "input_dim",
        "output_dim",
        "targets",
    }

    missing_keys = sorted(required_keys.difference(metadata))

    if missing_keys:
        raise ValueError(f"Neural ONNX metadata is missing required keys: {missing_keys}")

    return metadata


def _create_cpu_session(
    model_path: Path,
) -> ort.InferenceSession:
    """Create a CPU ONNX Runtime inference session."""
    if not model_path.exists():
        raise FileNotFoundError(f"Neural ONNX model does not exist: {model_path}")

    available_providers = ort.get_available_providers()

    if "CPUExecutionProvider" not in available_providers:
        raise RuntimeError("CPUExecutionProvider is not available in ONNX Runtime.")

    return ort.InferenceSession(
        str(model_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )


@dataclass
class NeuralOnnxSurrogate:
    """Deployable ONNX Runtime neural surrogate."""

    preprocessor: NeuralPreprocessor
    targets: tuple[str, ...]
    input_name: str
    output_name: str
    session: ort.InferenceSession

    @classmethod
    def load(
        cls,
        model_path: Path,
        metadata_path: Path,
        preprocessing_path: Path,
    ) -> NeuralOnnxSurrogate:
        """Load ONNX graph, metadata, and preprocessing state."""
        metadata = _load_neural_onnx_metadata(metadata_path)

        preprocessor = NeuralPreprocessor.load(preprocessing_path)

        targets = tuple(str(target) for target in metadata["targets"])

        input_dim = int(metadata["input_dim"])

        output_dim = int(metadata["output_dim"])

        if preprocessor.input_dim != input_dim:
            raise ValueError("Preprocessor input dimension does not match ONNX metadata.")

        if preprocessor.output_dim != output_dim:
            raise ValueError("Preprocessor output dimension does not match ONNX metadata.")

        if tuple(preprocessor.targets) != targets:
            raise ValueError("Preprocessor targets do not match ONNX metadata.")

        if len(targets) != output_dim:
            raise ValueError("ONNX metadata target count does not match output_dim.")

        session = _create_cpu_session(model_path)

        session_inputs = session.get_inputs()

        session_outputs = session.get_outputs()

        if len(session_inputs) != 1:
            raise RuntimeError("Expected exactly one neural ONNX input.")

        if len(session_outputs) != 1:
            raise RuntimeError("Expected exactly one neural ONNX output.")

        input_name = str(metadata["input_name"])

        output_name = str(metadata["output_name"])

        if session_inputs[0].name != input_name:
            raise ValueError("ONNX Runtime input name does not match metadata.")

        if session_outputs[0].name != output_name:
            raise ValueError("ONNX Runtime output name does not match metadata.")

        return cls(
            preprocessor=preprocessor,
            targets=targets,
            input_name=input_name,
            output_name=output_name,
            session=session,
        )

    def predict_normalized(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Predict normalized targets with ONNX Runtime."""
        features = self.preprocessor.transform_features(frame)

        raw_outputs = self.session.run(
            [
                self.output_name,
            ],
            {
                self.input_name: features,
            },
        )

        if len(raw_outputs) != 1:
            raise RuntimeError("Neural ONNX graph returned an unexpected number of outputs.")

        predictions = np.asarray(
            raw_outputs[0],
            dtype=np.float32,
        )

        expected_shape = (
            len(frame),
            len(self.targets),
        )

        if predictions.shape != expected_shape:
            raise RuntimeError(
                "Unexpected neural ONNX output "
                f"shape: {predictions.shape}; "
                f"expected {expected_shape}."
            )

        return predictions

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict neural-surrogate targets in physical units."""
        normalized_predictions = self.predict_normalized(frame)

        physical_predictions = self.preprocessor.inverse_transform_targets(normalized_predictions)

        return pd.DataFrame(
            physical_predictions,
            columns=self.targets,
            index=frame.index,
        )

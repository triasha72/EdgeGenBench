"""ONNX Runtime inference for EdgeGenBench models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from edgegenbench.deployment.feature_encoder import (
    EdgeFeatureEncoder,
)


def _load_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load shared ONNX export metadata."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"ONNX metadata does not exist: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(metadata, dict):
        raise ValueError("ONNX metadata must contain an object.")

    return metadata


def _create_session(
    model_path: Path,
) -> ort.InferenceSession:
    """Create a CPU ONNX Runtime session."""
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    available_providers = ort.get_available_providers()

    if "CPUExecutionProvider" in available_providers:
        providers = ["CPUExecutionProvider"]
    else:
        providers = available_providers

    return ort.InferenceSession(
        str(model_path),
        providers=providers,
    )


@dataclass
class OnnxSurrogate:
    """ONNX Runtime surrogate-model wrapper."""

    encoder: EdgeFeatureEncoder
    targets: tuple[str, ...]
    input_name: str
    session: ort.InferenceSession

    @classmethod
    def load(
        cls,
        model_path: Path,
        metadata_path: Path,
    ) -> OnnxSurrogate:
        """Load an exported surrogate model."""
        metadata = _load_metadata(metadata_path)

        encoder = EdgeFeatureEncoder.from_metadata(metadata["feature_encoder"])

        session = _create_session(model_path)

        session_input_name = session.get_inputs()[0].name

        expected_input_name = str(metadata["input_name"])

        if session_input_name != expected_input_name:
            raise ValueError("ONNX input name does not match metadata.")

        return cls(
            encoder=encoder,
            targets=tuple(str(target) for target in metadata["surrogate"]["targets"]),
            input_name=session_input_name,
            session=session,
        )

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict surrogate targets with ONNX Runtime."""
        encoded_values = self.encoder.transform(frame)

        raw_outputs = self.session.run(
            None,
            {self.input_name: (encoded_values)},
        )

        if not raw_outputs:
            raise RuntimeError("The surrogate ONNX graph returned no outputs.")

        predictions = np.asarray(
            raw_outputs[0],
            dtype=np.float64,
        )

        if predictions.ndim == 1:
            predictions = predictions.reshape(
                -1,
                1,
            )

        expected_shape = (
            len(frame),
            len(self.targets),
        )

        if predictions.shape != expected_shape:
            raise RuntimeError(
                "Unexpected surrogate ONNX output shape: "
                f"{predictions.shape}; expected "
                f"{expected_shape}."
            )

        return pd.DataFrame(
            predictions,
            columns=self.targets,
            index=frame.index,
        )


@dataclass
class OnnxFeasibilityClassifier:
    """ONNX Runtime feasibility-classifier wrapper."""

    encoder: EdgeFeatureEncoder
    threshold: float
    class_labels: tuple[int, ...]
    feasible_class_index: int
    input_name: str
    session: ort.InferenceSession

    @classmethod
    def load(
        cls,
        model_path: Path,
        metadata_path: Path,
    ) -> OnnxFeasibilityClassifier:
        """Load an exported feasibility classifier."""
        metadata = _load_metadata(metadata_path)

        encoder = EdgeFeatureEncoder.from_metadata(metadata["feature_encoder"])

        feasibility_metadata = metadata["feasibility"]

        session = _create_session(model_path)

        session_input_name = session.get_inputs()[0].name

        expected_input_name = str(metadata["input_name"])

        if session_input_name != expected_input_name:
            raise ValueError("ONNX input name does not match metadata.")

        return cls(
            encoder=encoder,
            threshold=float(feasibility_metadata["threshold"]),
            class_labels=tuple(
                int(class_label) for class_label in (feasibility_metadata["class_labels"])
            ),
            feasible_class_index=int(feasibility_metadata["feasible_class_index"]),
            input_name=session_input_name,
            session=session,
        )

    def predict_feasibility_probability(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Predict feasibility probability with ONNX Runtime."""
        encoded_values = self.encoder.transform(frame)

        raw_outputs = self.session.run(
            None,
            {self.input_name: (encoded_values)},
        )

        probability_outputs = [
            np.asarray(output)
            for output in raw_outputs
            if (
                np.asarray(output).ndim == 2
                and np.asarray(output).shape[0] == len(frame)
                and np.asarray(output).shape[1] == len(self.class_labels)
            )
        ]

        if len(probability_outputs) != 1:
            output_shapes = [np.asarray(output).shape for output in raw_outputs]

            raise RuntimeError(
                "Could not identify a unique classifier "
                "probability output. Output shapes: "
                f"{output_shapes}"
            )

        probabilities = np.asarray(
            probability_outputs[0],
            dtype=np.float64,
        )

        feasible_probability = probabilities[
            :,
            self.feasible_class_index,
        ]

        probability_tolerance = 1.0e-5

        minimum_probability = float(np.min(feasible_probability))

        maximum_probability = float(np.max(feasible_probability))

        if (
            minimum_probability < -probability_tolerance
            or maximum_probability > 1.0 + probability_tolerance
        ):
            raise RuntimeError(
                "ONNX feasibility probabilities are "
                "substantially outside zero to one: "
                f"minimum={minimum_probability:.8f}, "
                f"maximum={maximum_probability:.8f}."
            )

        feasible_probability = np.clip(
            feasible_probability,
            0.0,
            1.0,
        )

        return pd.Series(
            feasible_probability,
            index=frame.index,
            name="feasibility_probability",
            dtype=np.float64,
        )

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Apply the stored safety threshold."""
        probabilities = self.predict_feasibility_probability(frame)

        return pd.Series(
            probabilities >= self.threshold,
            index=frame.index,
            name="predicted_is_feasible",
            dtype=bool,
        )

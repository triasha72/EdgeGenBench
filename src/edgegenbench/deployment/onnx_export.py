"""Export trained EdgeGenBench estimators to ONNX."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
)

from edgegenbench.deployment.feature_encoder import (
    EdgeFeatureEncoder,
)
from edgegenbench.models.feasibility import (
    FeasibilityClassifier,
)
from edgegenbench.models.tree_surrogate import (
    TreeSurrogate,
)

DEFAULT_TARGET_OPSET = 17
ONNX_INPUT_NAME = "features"


@dataclass(frozen=True)
class EdgeExportArtifacts:
    """Files created during ONNX export."""

    surrogate_onnx_path: Path
    feasibility_onnx_path: Path
    metadata_path: Path
    feature_count: int
    surrogate_target_count: int
    feasibility_threshold: float


def _export_estimator(
    estimator: Any,
    feature_count: int,
    output_path: Path,
    model_name: str,
    options: dict[int, dict[str, Any]] | None = None,
    target_opset: int = DEFAULT_TARGET_OPSET,
    output_width: int | None = None,
) -> onnx.ModelProto:
    """Convert and save one fitted Scikit-learn estimator."""
    initial_types = [
        (
            ONNX_INPUT_NAME,
            FloatTensorType(
                [
                    None,
                    feature_count,
                ]
            ),
        )
    ]

    onnx_model = convert_sklearn(
        estimator,
        name=model_name,
        initial_types=initial_types,
        target_opset=target_opset,
        options=options,
    )

    if output_width is not None:
        if output_width < 1:
            raise ValueError("ONNX output width must be positive.")

        if len(onnx_model.graph.output) != 1:
            raise RuntimeError("Expected one surrogate ONNX output.")

        output_shape = onnx_model.graph.output[0].type.tensor_type.shape

        if len(output_shape.dim) != 2:
            raise RuntimeError("Expected a two-dimensional surrogate output.")

        output_shape.dim[1].dim_value = output_width

    onnx.checker.check_model(onnx_model)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(onnx_model.SerializeToString())

    return onnx_model


def _output_names(
    onnx_model: onnx.ModelProto,
) -> list[str]:
    """Return output names from an ONNX graph."""
    return [str(output.name) for output in onnx_model.graph.output]


def export_edge_models(
    surrogate_model_path: Path,
    feasibility_model_path: Path,
    output_dir: Path = Path("artifacts/edge_export"),
    target_opset: int = DEFAULT_TARGET_OPSET,
) -> EdgeExportArtifacts:
    """Export surrogate and feasibility estimators to ONNX."""
    if not surrogate_model_path.exists():
        raise FileNotFoundError(f"Surrogate model does not exist: {surrogate_model_path}")

    if not feasibility_model_path.exists():
        raise FileNotFoundError(f"Feasibility model does not exist: {feasibility_model_path}")

    surrogate_model = TreeSurrogate.load(surrogate_model_path)

    feasibility_model = FeasibilityClassifier.load(feasibility_model_path)

    surrogate_preprocessor = surrogate_model.pipeline.named_steps["preprocessor"]

    feasibility_preprocessor = feasibility_model.pipeline.named_steps["preprocessor"]

    surrogate_encoder = EdgeFeatureEncoder.from_fitted_preprocessor(surrogate_preprocessor)

    feasibility_encoder = EdgeFeatureEncoder.from_fitted_preprocessor(feasibility_preprocessor)

    if surrogate_encoder.to_metadata() != feasibility_encoder.to_metadata():
        raise ValueError("Surrogate and classifier feature encoders are not identical.")

    encoder = surrogate_encoder

    surrogate_estimator = surrogate_model.pipeline.named_steps["estimator"]

    classifier_estimator = feasibility_model.pipeline.named_steps["classifier"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    surrogate_onnx_path = output_dir / "surrogate.onnx"
    feasibility_onnx_path = output_dir / "feasibility.onnx"
    metadata_path = output_dir / "metadata.json"

    surrogate_onnx = _export_estimator(
        estimator=surrogate_estimator,
        feature_count=encoder.feature_count,
        output_path=surrogate_onnx_path,
        model_name="edgegenbench_surrogate",
        target_opset=target_opset,
        output_width=len(surrogate_model.targets),
    )

    classifier_options = {
        id(classifier_estimator): {
            "zipmap": False,
        }
    }

    feasibility_onnx = _export_estimator(
        estimator=classifier_estimator,
        feature_count=encoder.feature_count,
        output_path=feasibility_onnx_path,
        model_name=("edgegenbench_feasibility_classifier"),
        options=classifier_options,
        target_opset=target_opset,
    )

    class_labels = [
        int(class_label) for class_label in np.asarray(classifier_estimator.classes_).tolist()
    ]

    feasible_positions = [
        class_index for class_index, class_label in enumerate(class_labels) if class_label == 1
    ]

    if len(feasible_positions) != 1:
        raise RuntimeError("Classifier must contain one feasible class.")

    metadata = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "target_opset": target_opset,
        "input_name": ONNX_INPUT_NAME,
        "feature_encoder": (encoder.to_metadata()),
        "surrogate": {
            "source_model_path": str(surrogate_model_path),
            "onnx_model_path": str(surrogate_onnx_path),
            "targets": list(surrogate_model.targets),
            "output_names": _output_names(surrogate_onnx),
            "onnx_size_bytes": (surrogate_onnx_path.stat().st_size),
        },
        "feasibility": {
            "source_model_path": str(feasibility_model_path),
            "onnx_model_path": str(feasibility_onnx_path),
            "threshold": float(feasibility_model.threshold),
            "class_labels": class_labels,
            "feasible_class_index": int(feasible_positions[0]),
            "output_names": _output_names(feasibility_onnx),
            "onnx_size_bytes": (feasibility_onnx_path.stat().st_size),
        },
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

    return EdgeExportArtifacts(
        surrogate_onnx_path=(surrogate_onnx_path),
        feasibility_onnx_path=(feasibility_onnx_path),
        metadata_path=metadata_path,
        feature_count=encoder.feature_count,
        surrogate_target_count=len(surrogate_model.targets),
        feasibility_threshold=float(feasibility_model.threshold),
    )

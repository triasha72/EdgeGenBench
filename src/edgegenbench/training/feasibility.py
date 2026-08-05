"""Training pipeline for aircraft-design feasibility classification."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from edgegenbench.evaluation.classification import (
    build_confusion_matrix,
    build_probability_calibration,
    calculate_classification_metrics,
)
from edgegenbench.models.feasibility import (
    FEASIBILITY_TARGET,
    FeasibilityClassifier,
)
from edgegenbench.models.preprocessing import (
    FEATURE_COLUMNS,
)

DEFAULT_THRESHOLD_GRID = (
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)


@dataclass(frozen=True)
class FeasibilityArtifacts:
    """Artifacts created by feasibility-classifier training."""

    model_path: Path
    threshold_search_path: Path
    test_predictions_path: Path
    test_metrics_path: Path
    confusion_matrix_path: Path
    probability_calibration_path: Path
    summary_path: Path
    selected_threshold: float
    false_safe_rate: float
    balanced_accuracy: float
    test_rows: int


def _load_dataset(
    dataset_path: Path,
) -> pd.DataFrame:
    """Load and validate a feasibility-classification dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    required_columns = {
        *FEATURE_COLUMNS,
        FEASIBILITY_TARGET,
        "split",
    }

    missing_columns = sorted(required_columns.difference(frame.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if frame.loc[:, list(required_columns)].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    available_splits = set(frame["split"].astype(str))

    required_splits = {
        "train",
        "validation",
        "test",
    }

    missing_splits = sorted(required_splits.difference(available_splits))

    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")

    target_values = frame[FEASIBILITY_TARGET].astype(np.int64)

    available_classes = set(target_values.unique().tolist())

    if not available_classes.issubset({0, 1}):
        raise ValueError("Feasibility targets must be binary.")

    return frame


def _validate_threshold_grid(
    thresholds: Sequence[float],
) -> tuple[float, ...]:
    """Validate and normalize a threshold grid."""
    threshold_values = tuple(sorted({float(threshold) for threshold in thresholds}))

    if not threshold_values:
        raise ValueError("At least one threshold must be supplied.")

    if any(threshold < 0.0 or threshold > 1.0 for threshold in threshold_values):
        raise ValueError("Threshold values must be between zero and one.")

    return threshold_values


def select_feasibility_threshold(
    actual: Sequence[bool | int],
    probability: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    max_false_safe_rate: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """Select a safety-conscious feasibility threshold.

    Thresholds satisfying the maximum false-safe rate are ranked
    using balanced accuracy and feasible-design recall.

    When no threshold satisfies the safety target, the threshold
    with the lowest false-safe rate is selected.
    """
    if not 0.0 <= max_false_safe_rate <= 1.0:
        raise ValueError("max_false_safe_rate must be between zero and one.")

    threshold_values = _validate_threshold_grid(thresholds)

    records: list[dict[str, float | int]] = []

    for threshold in threshold_values:
        metrics = calculate_classification_metrics(
            actual=actual,
            probability=probability,
            threshold=threshold,
        )

        records.append(metrics)

    search_results = pd.DataFrame(records)

    false_safe_values = search_results["false_safe_rate"]

    search_results["meets_false_safe_target"] = false_safe_values.notna() & (
        false_safe_values <= max_false_safe_rate
    )

    eligible_results = search_results.loc[search_results["meets_false_safe_target"]].copy()

    if not eligible_results.empty:
        ranked_results = eligible_results.sort_values(
            by=[
                "balanced_accuracy",
                "recall",
                "precision",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            ignore_index=True,
        )
    else:
        fallback_results = search_results.copy()

        fallback_results["_false_safe_sort"] = fallback_results["false_safe_rate"].fillna(
            float("inf")
        )

        ranked_results = fallback_results.sort_values(
            by=[
                "_false_safe_sort",
                "balanced_accuracy",
                "recall",
                "threshold",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
            ignore_index=True,
        )

    selected_threshold = float(ranked_results.loc[0, "threshold"])

    search_results["selected"] = np.isclose(
        search_results["threshold"],
        selected_threshold,
    )

    search_results = search_results.sort_values(
        by="threshold",
        ignore_index=True,
    )

    return selected_threshold, search_results


def _create_prediction_report(
    test_frame: pd.DataFrame,
    probability: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Create a row-level held-out test prediction report."""
    report = (
        test_frame.loc[
            :,
            [
                *FEATURE_COLUMNS,
                FEASIBILITY_TARGET,
            ],
        ]
        .reset_index(drop=True)
        .copy()
    )

    probability_values = probability.reset_index(drop=True)

    report["feasibility_probability"] = probability_values

    report["infeasibility_probability"] = 1.0 - probability_values

    report["predicted_is_feasible"] = probability_values >= threshold

    report["selected_threshold"] = threshold

    report["false_safe_prediction"] = (
        ~report[FEASIBILITY_TARGET].astype(bool) & report["predicted_is_feasible"]
    )

    report["false_reject_prediction"] = (
        report[FEASIBILITY_TARGET].astype(bool) & ~report["predicted_is_feasible"]
    )

    return report


def _json_safe(
    value: Any,
) -> Any:
    """Convert NumPy and nonfinite values for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

        return value

    return value


def train_feasibility_classifier(
    dataset_path: Path,
    output_dir: Path = Path("artifacts/feasibility_classifier"),
    classifier_parameters: Mapping[
        str,
        Any,
    ]
    | None = None,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    max_false_safe_rate: float = 0.05,
    random_state: int = 42,
) -> FeasibilityArtifacts:
    """Train, select, evaluate, and save the classifier."""
    frame = _load_dataset(dataset_path)

    training_frame = frame.loc[frame["split"] == "train"].copy()

    validation_frame = frame.loc[frame["split"] == "validation"].copy()

    test_frame = frame.loc[frame["split"] == "test"].copy()

    selection_model = FeasibilityClassifier.fit(
        frame=training_frame,
        parameters=classifier_parameters,
        random_state=random_state,
    )

    validation_probability = selection_model.predict_feasibility_probability(validation_frame)

    selected_threshold, threshold_search = select_feasibility_threshold(
        actual=validation_frame[FEASIBILITY_TARGET].astype(np.int64),
        probability=validation_probability,
        thresholds=threshold_grid,
        max_false_safe_rate=(max_false_safe_rate),
    )

    final_training_frame = pd.concat(
        [
            training_frame,
            validation_frame,
        ],
        ignore_index=True,
    )

    training_start = perf_counter()

    final_model = FeasibilityClassifier.fit(
        frame=final_training_frame,
        parameters=classifier_parameters,
        threshold=selected_threshold,
        random_state=random_state,
    )

    final_training_time_seconds = perf_counter() - training_start

    test_probability = final_model.predict_feasibility_probability(test_frame)

    test_metrics = calculate_classification_metrics(
        actual=test_frame[FEASIBILITY_TARGET].astype(np.int64),
        probability=test_probability,
        threshold=selected_threshold,
    )

    confusion = build_confusion_matrix(
        actual=test_frame[FEASIBILITY_TARGET].astype(np.int64),
        probability=test_probability,
        threshold=selected_threshold,
    )

    probability_calibration = build_probability_calibration(
        actual=test_frame[FEASIBILITY_TARGET].astype(np.int64),
        probability=test_probability,
        bin_count=10,
    )

    prediction_report = _create_prediction_report(
        test_frame=test_frame,
        probability=test_probability,
        threshold=selected_threshold,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / "model.joblib"
    threshold_search_path = output_dir / "threshold_search.csv"
    test_predictions_path = output_dir / "test_predictions.csv"
    test_metrics_path = output_dir / "test_metrics.json"
    confusion_matrix_path = output_dir / "confusion_matrix.csv"
    probability_calibration_path = output_dir / "probability_calibration.csv"
    summary_path = output_dir / "summary.json"

    final_model.save(model_path)

    threshold_search.to_csv(
        threshold_search_path,
        index=False,
    )

    prediction_report.to_csv(
        test_predictions_path,
        index=False,
    )

    confusion.to_csv(
        confusion_matrix_path,
    )

    probability_calibration.to_csv(
        probability_calibration_path,
        index=False,
    )

    test_metrics_path.write_text(
        json.dumps(
            _json_safe(test_metrics),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "dataset_path": str(dataset_path),
        "model_type": ("Random Forest feasibility classifier"),
        "classifier_parameters": (final_model.parameters),
        "random_state": random_state,
        "selected_threshold": (selected_threshold),
        "max_false_safe_rate": (max_false_safe_rate),
        "training_rows": int(len(training_frame)),
        "validation_rows": int(len(validation_frame)),
        "final_training_rows": int(len(final_training_frame)),
        "test_rows": int(len(test_frame)),
        "final_training_time_seconds": (final_training_time_seconds),
        "model_size_bytes": int(model_path.stat().st_size),
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "threshold_search_path": str(threshold_search_path),
        "test_predictions_path": str(test_predictions_path),
        "test_metrics_path": str(test_metrics_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "probability_calibration_path": str(probability_calibration_path),
    }

    summary_path.write_text(
        json.dumps(
            _json_safe(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return FeasibilityArtifacts(
        model_path=model_path,
        threshold_search_path=(threshold_search_path),
        test_predictions_path=(test_predictions_path),
        test_metrics_path=test_metrics_path,
        confusion_matrix_path=(confusion_matrix_path),
        probability_calibration_path=(probability_calibration_path),
        summary_path=summary_path,
        selected_threshold=(selected_threshold),
        false_safe_rate=float(test_metrics["false_safe_rate"]),
        balanced_accuracy=float(test_metrics["balanced_accuracy"]),
        test_rows=len(test_frame),
    )

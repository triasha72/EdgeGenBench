"""Training pipelines for nonlinear EdgeGenBench baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from edgegenbench.evaluation.regression import (
    calculate_regression_metrics,
)
from edgegenbench.models.fp32_linear import DEFAULT_TARGETS
from edgegenbench.models.preprocessing import FEATURE_COLUMNS
from edgegenbench.models.tree_surrogate import (
    HIST_GRADIENT_BOOSTING,
    RANDOM_FOREST,
    SUPPORTED_MODEL_TYPES,
    TreeSurrogate,
)

DEFAULT_MODEL_GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    RANDOM_FOREST: (
        {
            "n_estimators": 150,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 250,
            "max_depth": 16,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 250,
            "max_depth": 24,
            "min_samples_leaf": 1,
        },
    ),
    HIST_GRADIENT_BOOSTING: (
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 31,
        },
        {
            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 31,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 63,
        },
    ),
}


@dataclass(frozen=True)
class TreeBaselineArtifacts:
    """Artifacts created for one nonlinear model family."""

    model_type: str
    model_path: Path
    validation_search_path: Path
    test_metrics_path: Path
    test_predictions_path: Path
    latency_path: Path
    summary_path: Path
    best_parameters: dict[str, Any]
    mean_test_nrmse_std: float
    mean_test_r2: float


def _load_dataset(
    dataset_path: Path,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Load and validate the benchmark dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    required_columns = {
        *FEATURE_COLUMNS,
        *targets,
        "split",
    }
    missing_columns = sorted(required_columns.difference(frame.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    expected_splits = {"train", "validation", "test"}
    available_splits = set(frame["split"].astype(str))
    missing_splits = sorted(expected_splits.difference(available_splits))

    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")

    return frame


def _select_configuration(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    model_type: str,
    targets: tuple[str, ...],
    configurations: Sequence[Mapping[str, Any]],
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select model parameters using validation NRMSE."""
    if not configurations:
        raise ValueError(f"No parameter configurations supplied for {model_type}.")

    records: list[dict[str, Any]] = []

    for candidate_index, parameters in enumerate(configurations):
        start_time = perf_counter()

        model = TreeSurrogate.fit(
            frame=training_frame,
            model_type=model_type,
            targets=targets,
            parameters=parameters,
            random_state=random_state,
        )

        training_time_seconds = perf_counter() - start_time

        predictions = model.predict(validation_frame)
        metrics = calculate_regression_metrics(
            actual=validation_frame,
            predicted=predictions,
            targets=targets,
        )

        records.append(
            {
                "candidate_index": candidate_index,
                "parameters": json.dumps(
                    dict(parameters),
                    sort_keys=True,
                ),
                "mean_validation_nrmse_std": float(np.nanmean(metrics["nrmse_std"])),
                "mean_validation_r2": float(np.nanmean(metrics["r2"])),
                "training_time_seconds": training_time_seconds,
            }
        )

    search_results = pd.DataFrame(records).sort_values(
        by=[
            "mean_validation_nrmse_std",
            "candidate_index",
        ],
        ascending=[True, True],
        ignore_index=True,
    )

    best_candidate_index = int(search_results.loc[0, "candidate_index"])
    best_parameters = dict(configurations[best_candidate_index])

    return best_parameters, search_results


def _create_prediction_report(
    test_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    targets: tuple[str, ...],
) -> pd.DataFrame:
    """Combine inputs, actual targets, and predictions."""
    report = test_frame.loc[
        :,
        [*FEATURE_COLUMNS, *targets],
    ].reset_index(drop=True)

    for target in targets:
        report[f"predicted_{target}"] = predictions[target].to_numpy()

    return report


def _benchmark_latency(
    model: TreeSurrogate,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Measure prediction latency at several batch sizes."""
    records: list[dict[str, float | int]] = []

    for batch_size in (1, 32, 256):
        if len(test_frame) < batch_size:
            continue

        batch = test_frame.iloc[:batch_size].copy()
        repeats = 30 if batch_size == 1 else 10

        for _ in range(3):
            model.predict(batch)

        start_time = perf_counter()

        for _ in range(repeats):
            model.predict(batch)

        elapsed_seconds = perf_counter() - start_time
        mean_batch_latency_ms = (elapsed_seconds / repeats) * 1000.0

        records.append(
            {
                "batch_size": batch_size,
                "repeats": repeats,
                "mean_batch_latency_ms": mean_batch_latency_ms,
                "mean_sample_latency_us": (mean_batch_latency_ms * 1000.0 / batch_size),
            }
        )

    return pd.DataFrame(records)


def _train_one_model(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    model_type: str,
    targets: tuple[str, ...],
    configurations: Sequence[Mapping[str, Any]],
    output_root: Path,
    random_state: int,
) -> TreeBaselineArtifacts:
    """Train, select, evaluate, and save one model family."""
    best_parameters, validation_search = _select_configuration(
        training_frame=training_frame,
        validation_frame=validation_frame,
        model_type=model_type,
        targets=targets,
        configurations=configurations,
        random_state=random_state,
    )

    final_training_frame = pd.concat(
        [training_frame, validation_frame],
        ignore_index=True,
    )

    training_start = perf_counter()

    final_model = TreeSurrogate.fit(
        frame=final_training_frame,
        model_type=model_type,
        targets=targets,
        parameters=best_parameters,
        random_state=random_state,
    )

    final_training_time_seconds = perf_counter() - training_start

    test_predictions = final_model.predict(test_frame)

    test_metrics = calculate_regression_metrics(
        actual=test_frame,
        predicted=test_predictions,
        targets=targets,
    )

    prediction_report = _create_prediction_report(
        test_frame=test_frame,
        predictions=test_predictions,
        targets=targets,
    )

    latency_report = _benchmark_latency(
        model=final_model,
        test_frame=test_frame,
    )

    output_dir = output_root / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    validation_search_path = output_dir / "validation_search.csv"
    test_metrics_path = output_dir / "test_metrics.csv"
    test_predictions_path = output_dir / "test_predictions.csv"
    latency_path = output_dir / "latency.csv"
    summary_path = output_dir / "summary.json"

    final_model.save(model_path)
    validation_search.to_csv(validation_search_path, index=False)
    test_metrics.to_csv(test_metrics_path, index=False)
    prediction_report.to_csv(test_predictions_path, index=False)
    latency_report.to_csv(latency_path, index=False)

    mean_test_nrmse_std = float(np.nanmean(test_metrics["nrmse_std"]))
    mean_test_r2 = float(np.nanmean(test_metrics["r2"]))

    summary = {
        "model_type": model_type,
        "best_parameters": best_parameters,
        "random_state": random_state,
        "targets": list(targets),
        "training_rows": int(len(training_frame)),
        "validation_rows": int(len(validation_frame)),
        "final_training_rows": int(len(final_training_frame)),
        "test_rows": int(len(test_frame)),
        "final_training_time_seconds": final_training_time_seconds,
        "mean_test_nrmse_std": mean_test_nrmse_std,
        "mean_test_r2": mean_test_r2,
        "model_size_bytes": int(model_path.stat().st_size),
        "model_path": str(model_path),
        "validation_search_path": str(validation_search_path),
        "test_metrics_path": str(test_metrics_path),
        "test_predictions_path": str(test_predictions_path),
        "latency_path": str(latency_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return TreeBaselineArtifacts(
        model_type=model_type,
        model_path=model_path,
        validation_search_path=validation_search_path,
        test_metrics_path=test_metrics_path,
        test_predictions_path=test_predictions_path,
        latency_path=latency_path,
        summary_path=summary_path,
        best_parameters=best_parameters,
        mean_test_nrmse_std=mean_test_nrmse_std,
        mean_test_r2=mean_test_r2,
    )


def train_tree_baselines(
    dataset_path: Path,
    output_dir: Path = Path("artifacts/tree_baselines"),
    targets: Sequence[str] = DEFAULT_TARGETS,
    model_grids: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ]
    | None = None,
    random_state: int = 42,
) -> tuple[TreeBaselineArtifacts, ...]:
    """Train all configured nonlinear baseline families."""
    target_names = tuple(targets)

    if not target_names:
        raise ValueError("At least one target must be supplied.")

    frame = _load_dataset(
        dataset_path=dataset_path,
        targets=target_names,
    )

    training_frame = frame.loc[frame["split"] == "train"].copy()
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    test_frame = frame.loc[frame["split"] == "test"].copy()

    grids = model_grids or DEFAULT_MODEL_GRIDS

    unsupported_models = sorted(set(grids).difference(SUPPORTED_MODEL_TYPES))

    if unsupported_models:
        raise ValueError(f"Unsupported model grids supplied: {unsupported_models}")

    artifacts: list[TreeBaselineArtifacts] = []

    for model_type, configurations in grids.items():
        artifacts.append(
            _train_one_model(
                training_frame=training_frame,
                validation_frame=validation_frame,
                test_frame=test_frame,
                model_type=model_type,
                targets=target_names,
                configurations=configurations,
                output_root=output_dir,
                random_state=random_state,
            )
        )

    return tuple(artifacts)

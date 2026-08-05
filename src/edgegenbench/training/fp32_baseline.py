"""Training and evaluation pipeline for the FP32 linear baseline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from edgegenbench.evaluation.regression import calculate_regression_metrics
from edgegenbench.models.fp32_linear import (
    CATEGORICAL_FEATURE,
    DEFAULT_TARGETS,
    NUMERIC_FEATURES,
    FP32LinearSurrogate,
)

DEFAULT_ALPHA_GRID = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
)


@dataclass(frozen=True)
class FP32BaselineArtifacts:
    """Artifacts created by one FP32 baseline training run."""

    model_path: Path
    validation_search_path: Path
    test_metrics_path: Path
    test_predictions_path: Path
    latency_path: Path
    summary_path: Path
    best_alpha: float
    mean_test_nrmse_std: float
    mean_test_r2: float


def _load_dataset(
    dataset_path: Path,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Load and validate an EdgeGenBench dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    frame = pd.read_csv(dataset_path)

    required_columns = {
        *NUMERIC_FEATURES,
        CATEGORICAL_FEATURE,
        *targets,
        "split",
    }
    missing_columns = sorted(required_columns.difference(frame.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    available_splits = set(frame["split"].astype(str))
    required_splits = {"train", "validation", "test"}
    missing_splits = sorted(required_splits.difference(available_splits))

    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")

    return frame


def _select_alpha(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    targets: tuple[str, ...],
    alpha_grid: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    """Select the alpha with the lowest mean validation NRMSE."""
    records: list[dict[str, float]] = []

    for alpha in alpha_grid:
        if alpha <= 0.0:
            raise ValueError("Every alpha value must be greater than zero.")

        model = FP32LinearSurrogate.fit(
            frame=training_frame,
            targets=targets,
            alpha=float(alpha),
        )

        predictions = model.predict(validation_frame)
        metrics = calculate_regression_metrics(
            actual=validation_frame,
            predicted=predictions,
            targets=targets,
        )

        records.append(
            {
                "alpha": float(alpha),
                "mean_validation_nrmse_std": float(np.nanmean(metrics["nrmse_std"])),
                "mean_validation_r2": float(np.nanmean(metrics["r2"])),
            }
        )

    search_results = pd.DataFrame(records).sort_values(
        by=["mean_validation_nrmse_std", "alpha"],
        ascending=[True, True],
        ignore_index=True,
    )

    best_alpha = float(search_results.loc[0, "alpha"])

    return best_alpha, search_results


def _create_prediction_report(
    test_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    targets: tuple[str, ...],
) -> pd.DataFrame:
    """Combine test inputs, actual values, and model predictions."""
    report_columns = [
        *NUMERIC_FEATURES,
        CATEGORICAL_FEATURE,
        *targets,
    ]

    report = test_frame.loc[:, report_columns].reset_index(drop=True).copy()

    for target in targets:
        report[f"predicted_{target}"] = predictions[target].to_numpy()

    return report


def _benchmark_latency(
    model: FP32LinearSurrogate,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Measure Python FP32 prediction latency for several batch sizes."""
    records: list[dict[str, float | int]] = []

    for batch_size in (1, 32, 256):
        if len(test_frame) < batch_size:
            continue

        batch = test_frame.iloc[:batch_size].copy()
        repeats = 200 if batch_size == 1 else 100

        for _ in range(5):
            model.predict(batch)

        start_time = perf_counter()

        for _ in range(repeats):
            model.predict(batch)

        elapsed_seconds = perf_counter() - start_time
        mean_batch_latency_ms = (elapsed_seconds / repeats) * 1000.0

        mean_sample_latency_us = mean_batch_latency_ms * 1000.0 / batch_size

        records.append(
            {
                "batch_size": batch_size,
                "repeats": repeats,
                "mean_batch_latency_ms": mean_batch_latency_ms,
                "mean_sample_latency_us": mean_sample_latency_us,
            }
        )

    return pd.DataFrame(records)


def train_fp32_baseline(
    dataset_path: Path,
    output_dir: Path = Path("artifacts/fp32_baseline"),
    targets: Sequence[str] = DEFAULT_TARGETS,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> FP32BaselineArtifacts:
    """Train, select, evaluate, benchmark, and save the FP32 baseline."""
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

    best_alpha, validation_search = _select_alpha(
        training_frame=training_frame,
        validation_frame=validation_frame,
        targets=target_names,
        alpha_grid=alpha_grid,
    )

    final_training_frame = pd.concat(
        [training_frame, validation_frame],
        ignore_index=True,
    )

    final_model = FP32LinearSurrogate.fit(
        frame=final_training_frame,
        targets=target_names,
        alpha=best_alpha,
    )

    test_predictions = final_model.predict(test_frame)

    test_metrics = calculate_regression_metrics(
        actual=test_frame,
        predicted=test_predictions,
        targets=target_names,
    )

    prediction_report = _create_prediction_report(
        test_frame=test_frame,
        predictions=test_predictions,
        targets=target_names,
    )

    latency_report = _benchmark_latency(
        model=final_model,
        test_frame=test_frame,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "fp32_linear_model.npz"
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
        "dataset_path": str(dataset_path),
        "model_type": "FP32 multi-output ridge regression",
        "best_alpha": best_alpha,
        "targets": list(target_names),
        "training_rows": int(len(training_frame)),
        "validation_rows": int(len(validation_frame)),
        "final_training_rows": int(len(final_training_frame)),
        "test_rows": int(len(test_frame)),
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

    return FP32BaselineArtifacts(
        model_path=model_path,
        validation_search_path=validation_search_path,
        test_metrics_path=test_metrics_path,
        test_predictions_path=test_predictions_path,
        latency_path=latency_path,
        summary_path=summary_path,
        best_alpha=best_alpha,
        mean_test_nrmse_std=mean_test_nrmse_std,
        mean_test_r2=mean_test_r2,
    )

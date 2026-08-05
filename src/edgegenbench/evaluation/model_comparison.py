"""Unified comparison of EdgeGenBench surrogate models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edgegenbench.evaluation.plots import (
    plot_accuracy_vs_latency,
    plot_accuracy_vs_model_size,
    plot_mean_nrmse,
    plot_mean_r2,
    plot_target_nrmse_heatmap,
)

MODEL_DIRECTORIES = {
    "fp32_ridge": Path("fp32_baseline"),
    "random_forest": Path("tree_baselines/random_forest"),
    "hist_gradient_boosting": Path("tree_baselines/hist_gradient_boosting"),
}

REQUIRED_METRIC_COLUMNS = {
    "target",
    "mae",
    "rmse",
    "nrmse_std",
    "r2",
}

REQUIRED_LATENCY_COLUMNS = {
    "batch_size",
    "mean_batch_latency_ms",
}


@dataclass(frozen=True)
class ModelComparisonArtifacts:
    """Files and rankings produced by model comparison."""

    detailed_metrics_path: Path
    aggregate_metrics_path: Path
    latency_comparison_path: Path
    summary_path: Path
    plot_paths: tuple[Path, ...]
    best_accuracy_model: str
    best_mean_r2_model: str
    lowest_latency_model: str
    smallest_model: str


def _require_file(path: Path) -> None:
    """Raise a clear error when an artifact is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Required model artifact does not exist: {path}")


def _load_summary(path: Path) -> dict[str, Any]:
    """Load and validate one model summary."""
    _require_file(path)

    summary = json.loads(path.read_text(encoding="utf-8"))

    if "model_size_bytes" not in summary:
        raise ValueError(f"Summary is missing model_size_bytes: {path}")

    return summary


def _load_metrics(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    """Load target-level test metrics for one model."""
    _require_file(path)

    metrics = pd.read_csv(path)

    missing_columns = REQUIRED_METRIC_COLUMNS.difference(metrics.columns)

    if missing_columns:
        raise ValueError(f"Metrics file is missing columns {sorted(missing_columns)}: {path}")

    metrics = metrics.loc[
        :,
        [
            "target",
            "mae",
            "rmse",
            "nrmse_std",
            "r2",
        ],
    ].copy()

    metrics.insert(0, "model", model_name)

    return metrics


def _load_latency(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    """Load latency measurements for one model."""
    _require_file(path)

    latency = pd.read_csv(path)

    missing_columns = REQUIRED_LATENCY_COLUMNS.difference(latency.columns)

    if missing_columns:
        raise ValueError(f"Latency file is missing columns {sorted(missing_columns)}: {path}")

    latency = latency.copy()
    latency.insert(0, "model", model_name)

    return latency


def _get_batch_latency(
    latency: pd.DataFrame,
    batch_size: int,
) -> float:
    """Return mean latency for one batch size."""
    matching_rows = latency.loc[
        latency["batch_size"] == batch_size,
        "mean_batch_latency_ms",
    ]

    if matching_rows.empty:
        return float("nan")

    return float(matching_rows.iloc[0])


def _select_winner(
    aggregate_metrics: pd.DataFrame,
    metric: str,
    ascending: bool,
) -> str:
    """Select a deterministic winner for one metric."""
    valid_rows = aggregate_metrics.dropna(subset=[metric])

    if valid_rows.empty:
        return "not_available"

    ranked_rows = valid_rows.sort_values(
        by=[metric, "model"],
        ascending=[ascending, True],
        ignore_index=True,
    )

    return str(ranked_rows.loc[0, "model"])


def compare_model_artifacts(
    artifact_root: Path = Path("artifacts"),
    output_dir: Path = Path("reports/model_comparison"),
) -> ModelComparisonArtifacts:
    """Compare all completed surrogate-model runs."""
    detailed_frames: list[pd.DataFrame] = []
    latency_frames: list[pd.DataFrame] = []
    aggregate_records: list[dict[str, float | str]] = []

    for model_name, relative_directory in MODEL_DIRECTORIES.items():
        model_directory = artifact_root / relative_directory

        summary = _load_summary(model_directory / "summary.json")

        metrics = _load_metrics(
            path=model_directory / "test_metrics.csv",
            model_name=model_name,
        )

        latency = _load_latency(
            path=model_directory / "latency.csv",
            model_name=model_name,
        )

        detailed_frames.append(metrics)
        latency_frames.append(latency)

        model_size_bytes = float(summary["model_size_bytes"])

        training_time_seconds = float(
            summary.get(
                "final_training_time_seconds",
                float("nan"),
            )
        )

        aggregate_records.append(
            {
                "model": model_name,
                "mean_mae": float(np.nanmean(metrics["mae"])),
                "mean_rmse": float(np.nanmean(metrics["rmse"])),
                "mean_nrmse_std": float(np.nanmean(metrics["nrmse_std"])),
                "mean_r2": float(np.nanmean(metrics["r2"])),
                "model_size_bytes": model_size_bytes,
                "model_size_mb": (model_size_bytes / (1024.0**2)),
                "training_time_seconds": (training_time_seconds),
                "batch_1_latency_ms": _get_batch_latency(
                    latency,
                    batch_size=1,
                ),
                "batch_32_latency_ms": _get_batch_latency(
                    latency,
                    batch_size=32,
                ),
                "batch_256_latency_ms": (
                    _get_batch_latency(
                        latency,
                        batch_size=256,
                    )
                ),
            }
        )

    detailed_metrics = pd.concat(
        detailed_frames,
        ignore_index=True,
    ).sort_values(
        by=["model", "target"],
        ignore_index=True,
    )

    latency_comparison = pd.concat(
        latency_frames,
        ignore_index=True,
    ).sort_values(
        by=["model", "batch_size"],
        ignore_index=True,
    )

    aggregate_metrics = pd.DataFrame(aggregate_records).sort_values(
        by="mean_nrmse_std",
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_metrics_path = output_dir / "detailed_metrics.csv"
    aggregate_metrics_path = output_dir / "aggregate_metrics.csv"
    latency_comparison_path = output_dir / "latency_comparison.csv"
    summary_path = output_dir / "comparison_summary.json"

    detailed_metrics.to_csv(
        detailed_metrics_path,
        index=False,
    )
    aggregate_metrics.to_csv(
        aggregate_metrics_path,
        index=False,
    )
    latency_comparison.to_csv(
        latency_comparison_path,
        index=False,
    )

    best_accuracy_model = _select_winner(
        aggregate_metrics,
        metric="mean_nrmse_std",
        ascending=True,
    )

    best_mean_r2_model = _select_winner(
        aggregate_metrics,
        metric="mean_r2",
        ascending=False,
    )

    lowest_latency_model = _select_winner(
        aggregate_metrics,
        metric="batch_1_latency_ms",
        ascending=True,
    )

    smallest_model = _select_winner(
        aggregate_metrics,
        metric="model_size_bytes",
        ascending=True,
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "models_compared": list(aggregate_metrics["model"]),
        "best_accuracy_model": best_accuracy_model,
        "best_mean_r2_model": best_mean_r2_model,
        "lowest_batch_1_latency_model": (lowest_latency_model),
        "smallest_model": smallest_model,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_paths = (
        plot_mean_nrmse(
            aggregate_metrics,
            output_dir / "mean_nrmse_by_model.png",
        ),
        plot_mean_r2(
            aggregate_metrics,
            output_dir / "mean_r2_by_model.png",
        ),
        plot_accuracy_vs_latency(
            aggregate_metrics,
            output_dir / "accuracy_vs_latency.png",
        ),
        plot_accuracy_vs_model_size(
            aggregate_metrics,
            output_dir / "accuracy_vs_model_size.png",
        ),
        plot_target_nrmse_heatmap(
            detailed_metrics,
            output_dir / "target_nrmse_heatmap.png",
        ),
    )

    return ModelComparisonArtifacts(
        detailed_metrics_path=detailed_metrics_path,
        aggregate_metrics_path=aggregate_metrics_path,
        latency_comparison_path=(latency_comparison_path),
        summary_path=summary_path,
        plot_paths=plot_paths,
        best_accuracy_model=best_accuracy_model,
        best_mean_r2_model=best_mean_r2_model,
        lowest_latency_model=lowest_latency_model,
        smallest_model=smallest_model,
    )

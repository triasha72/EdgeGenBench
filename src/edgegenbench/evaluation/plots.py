"""Plots for comparing EdgeGenBench surrogate models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_output_path(output_path: Path) -> None:
    """Create the parent directory for a plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)


def plot_mean_nrmse(
    aggregate_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot mean normalized RMSE for every model."""
    required_columns = {"model", "mean_nrmse_std"}
    missing_columns = required_columns.difference(aggregate_metrics.columns)

    if missing_columns:
        raise ValueError(f"Aggregate metrics are missing columns: {sorted(missing_columns)}")

    _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        aggregate_metrics["model"],
        aggregate_metrics["mean_nrmse_std"],
    )

    axis.set_title("Mean Test NRMSE by Model")
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean normalized RMSE")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def plot_mean_r2(
    aggregate_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot mean test R-squared for every model."""
    required_columns = {"model", "mean_r2"}
    missing_columns = required_columns.difference(aggregate_metrics.columns)

    if missing_columns:
        raise ValueError(f"Aggregate metrics are missing columns: {sorted(missing_columns)}")

    _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        aggregate_metrics["model"],
        aggregate_metrics["mean_r2"],
    )

    axis.set_title("Mean Test R² by Model")
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean R²")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def plot_accuracy_vs_latency(
    aggregate_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot model accuracy against batch-one latency."""
    required_columns = {
        "model",
        "mean_nrmse_std",
        "batch_1_latency_ms",
    }
    missing_columns = required_columns.difference(aggregate_metrics.columns)

    if missing_columns:
        raise ValueError(f"Aggregate metrics are missing columns: {sorted(missing_columns)}")

    plotting_frame = aggregate_metrics.dropna(
        subset=[
            "mean_nrmse_std",
            "batch_1_latency_ms",
        ]
    )

    _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.scatter(
        plotting_frame["batch_1_latency_ms"],
        plotting_frame["mean_nrmse_std"],
        s=80,
    )

    for row in plotting_frame.itertuples(index=False):
        axis.annotate(
            row.model,
            (
                row.batch_1_latency_ms,
                row.mean_nrmse_std,
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    axis.set_title("Accuracy–Latency Trade-off")
    axis.set_xlabel("Batch-1 latency (ms)")
    axis.set_ylabel("Mean normalized RMSE")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def plot_accuracy_vs_model_size(
    aggregate_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot model accuracy against serialized model size."""
    required_columns = {
        "model",
        "mean_nrmse_std",
        "model_size_mb",
    }
    missing_columns = required_columns.difference(aggregate_metrics.columns)

    if missing_columns:
        raise ValueError(f"Aggregate metrics are missing columns: {sorted(missing_columns)}")

    plotting_frame = aggregate_metrics.dropna(
        subset=[
            "mean_nrmse_std",
            "model_size_mb",
        ]
    )

    _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.scatter(
        plotting_frame["model_size_mb"],
        plotting_frame["mean_nrmse_std"],
        s=80,
    )

    for row in plotting_frame.itertuples(index=False):
        axis.annotate(
            row.model,
            (
                row.model_size_mb,
                row.mean_nrmse_std,
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    axis.set_title("Accuracy–Model Size Trade-off")
    axis.set_xlabel("Model size (MB)")
    axis.set_ylabel("Mean normalized RMSE")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def plot_target_nrmse_heatmap(
    detailed_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot target-level normalized RMSE for all models."""
    required_columns = {
        "model",
        "target",
        "nrmse_std",
    }
    missing_columns = required_columns.difference(detailed_metrics.columns)

    if missing_columns:
        raise ValueError(f"Detailed metrics are missing columns: {sorted(missing_columns)}")

    pivot_table = detailed_metrics.pivot(
        index="target",
        columns="model",
        values="nrmse_std",
    )

    values = pivot_table.to_numpy(dtype=np.float64)

    _prepare_output_path(output_path)

    figure_width = max(8.0, 2.0 * len(pivot_table.columns))
    figure_height = max(5.0, 0.65 * len(pivot_table.index))

    figure, axis = plt.subplots(figsize=(figure_width, figure_height))

    image = axis.imshow(values, aspect="auto")

    axis.set_xticks(np.arange(len(pivot_table.columns)))
    axis.set_xticklabels(
        pivot_table.columns,
        rotation=20,
        ha="right",
    )

    axis.set_yticks(np.arange(len(pivot_table.index)))
    axis.set_yticklabels(pivot_table.index)

    axis.set_title("Target-Level Test NRMSE")
    axis.set_xlabel("Model")
    axis.set_ylabel("Target")

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Normalized RMSE")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path

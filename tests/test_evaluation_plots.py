"""Tests for model-comparison plots."""

from pathlib import Path

import pandas as pd

from edgegenbench.evaluation.plots import (
    plot_accuracy_vs_latency,
    plot_accuracy_vs_model_size,
    plot_mean_nrmse,
    plot_mean_r2,
    plot_target_nrmse_heatmap,
)


def _aggregate_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [
                "fp32_ridge",
                "random_forest",
                "hist_gradient_boosting",
            ],
            "mean_nrmse_std": [
                0.20,
                0.10,
                0.08,
            ],
            "mean_r2": [
                0.85,
                0.93,
                0.95,
            ],
            "batch_1_latency_ms": [
                0.10,
                0.80,
                0.60,
            ],
            "model_size_mb": [
                0.01,
                4.0,
                2.0,
            ],
        }
    )


def _detailed_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [
                "fp32_ridge",
                "random_forest",
                "hist_gradient_boosting",
            ]
            * 2,
            "target": [
                "mass",
                "mass",
                "mass",
                "energy",
                "energy",
                "energy",
            ],
            "nrmse_std": [
                0.20,
                0.10,
                0.08,
                0.18,
                0.09,
                0.07,
            ],
        }
    )


def _assert_nonempty_file(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_comparison_plots_are_created(
    tmp_path: Path,
) -> None:
    aggregate = _aggregate_metrics()
    detailed = _detailed_metrics()

    plot_paths = [
        plot_mean_nrmse(
            aggregate,
            tmp_path / "mean_nrmse.png",
        ),
        plot_mean_r2(
            aggregate,
            tmp_path / "mean_r2.png",
        ),
        plot_accuracy_vs_latency(
            aggregate,
            tmp_path / "accuracy_latency.png",
        ),
        plot_accuracy_vs_model_size(
            aggregate,
            tmp_path / "accuracy_size.png",
        ),
        plot_target_nrmse_heatmap(
            detailed,
            tmp_path / "heatmap.png",
        ),
    ]

    for path in plot_paths:
        _assert_nonempty_file(path)

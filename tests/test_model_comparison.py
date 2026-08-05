"""Tests for unified model comparison."""

import json
from pathlib import Path

import pandas as pd
import pytest

from edgegenbench.evaluation.model_comparison import (
    compare_model_artifacts,
)


def _write_model_artifacts(
    directory: Path,
    *,
    model_size_bytes: int,
    nrmse: float,
    r2: float,
    batch_1_latency_ms: float,
    training_time_seconds: float | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    summary: dict[str, float | int] = {
        "model_size_bytes": model_size_bytes,
        "mean_test_nrmse_std": nrmse,
        "mean_test_r2": r2,
    }

    if training_time_seconds is not None:
        summary["final_training_time_seconds"] = training_time_seconds

    (directory / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    metrics = pd.DataFrame(
        {
            "target": ["mass", "energy"],
            "mae": [10.0, 20.0],
            "rmse": [12.0, 24.0],
            "nrmse_std": [nrmse, nrmse],
            "r2": [r2, r2],
        }
    )

    metrics.to_csv(
        directory / "test_metrics.csv",
        index=False,
    )

    latency = pd.DataFrame(
        {
            "batch_size": [1, 32, 256],
            "repeats": [10, 10, 10],
            "mean_batch_latency_ms": [
                batch_1_latency_ms,
                batch_1_latency_ms * 3.0,
                batch_1_latency_ms * 10.0,
            ],
            "mean_sample_latency_us": [
                batch_1_latency_ms * 1000.0,
                batch_1_latency_ms * 100.0,
                batch_1_latency_ms * 40.0,
            ],
        }
    )

    latency.to_csv(
        directory / "latency.csv",
        index=False,
    )


def _create_artifact_tree(root: Path) -> None:
    _write_model_artifacts(
        root / "fp32_baseline",
        model_size_bytes=10_000,
        nrmse=0.20,
        r2=0.85,
        batch_1_latency_ms=0.10,
    )

    _write_model_artifacts(
        root / "tree_baselines/random_forest",
        model_size_bytes=4_000_000,
        nrmse=0.10,
        r2=0.93,
        batch_1_latency_ms=0.80,
        training_time_seconds=3.0,
    )

    _write_model_artifacts(
        root / "tree_baselines" / "hist_gradient_boosting",
        model_size_bytes=2_000_000,
        nrmse=0.08,
        r2=0.95,
        batch_1_latency_ms=0.60,
        training_time_seconds=2.0,
    )


def test_model_comparison_creates_outputs(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    output_dir = tmp_path / "reports"

    _create_artifact_tree(artifact_root)

    result = compare_model_artifacts(
        artifact_root=artifact_root,
        output_dir=output_dir,
    )

    assert result.best_accuracy_model == ("hist_gradient_boosting")
    assert result.best_mean_r2_model == ("hist_gradient_boosting")
    assert result.lowest_latency_model == "fp32_ridge"
    assert result.smallest_model == "fp32_ridge"

    assert result.detailed_metrics_path.exists()
    assert result.aggregate_metrics_path.exists()
    assert result.latency_comparison_path.exists()
    assert result.summary_path.exists()

    for plot_path in result.plot_paths:
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    aggregate = pd.read_csv(result.aggregate_metrics_path)

    assert len(aggregate) == 3
    assert set(aggregate["model"]) == {
        "fp32_ridge",
        "random_forest",
        "hist_gradient_boosting",
    }


def test_missing_artifacts_raise_clear_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Required model artifact",
    ):
        compare_model_artifacts(
            artifact_root=tmp_path / "missing",
            output_dir=tmp_path / "reports",
        )

"""Evaluation and plotting utilities for prediction intervals."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    frame_name: str,
) -> None:
    """Validate required DataFrame columns."""
    missing_columns = sorted(set(columns).difference(frame.columns))

    if missing_columns:
        raise ValueError(f"{frame_name} is missing columns: {missing_columns}")


def evaluate_prediction_intervals(
    actual: pd.DataFrame,
    intervals: pd.DataFrame,
    targets: Sequence[str],
    nominal_coverage: float,
    method: str,
) -> pd.DataFrame:
    """Evaluate interval coverage, width, and error relationships."""
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be between zero and one.")

    target_names = tuple(targets)

    if not target_names:
        raise ValueError("At least one target must be supplied.")

    _require_columns(
        frame=actual,
        columns=target_names,
        frame_name="Actual values",
    )

    records: list[dict[str, float | str]] = []

    for target in target_names:
        prediction_column = f"prediction_{target}"
        lower_column = f"lower_{target}"
        upper_column = f"upper_{target}"

        _require_columns(
            frame=intervals,
            columns=(
                prediction_column,
                lower_column,
                upper_column,
            ),
            frame_name="Prediction intervals",
        )

        actual_values = actual[target].to_numpy(dtype=np.float64)
        prediction_values = intervals[prediction_column].to_numpy(dtype=np.float64)
        lower_values = intervals[lower_column].to_numpy(dtype=np.float64)
        upper_values = intervals[upper_column].to_numpy(dtype=np.float64)

        if not (
            len(actual_values) == len(prediction_values) == len(lower_values) == len(upper_values)
        ):
            raise ValueError("Actual values and intervals must have identical row counts.")

        if not (
            np.isfinite(actual_values).all()
            and np.isfinite(prediction_values).all()
            and np.isfinite(lower_values).all()
            and np.isfinite(upper_values).all()
        ):
            raise ValueError("Actual values and prediction intervals must be finite.")

        interval_widths = upper_values - lower_values

        if np.any(interval_widths < 0.0):
            raise ValueError("Prediction interval upper bounds cannot be below lower bounds.")

        covered = (actual_values >= lower_values) & (actual_values <= upper_values)

        absolute_errors = np.abs(actual_values - prediction_values)

        empirical_coverage = float(np.mean(covered))
        mean_interval_width = float(np.mean(interval_widths))
        median_interval_width = float(np.median(interval_widths))

        target_scale = float(np.std(actual_values))

        if target_scale > np.finfo(np.float64).eps:
            normalized_interval_width = mean_interval_width / target_scale
        else:
            normalized_interval_width = float("nan")

        uncertainty_proxy = interval_widths / 2.0

        if (
            np.std(uncertainty_proxy) > np.finfo(np.float64).eps
            and np.std(absolute_errors) > np.finfo(np.float64).eps
        ):
            uncertainty_error_correlation = float(
                np.corrcoef(
                    uncertainty_proxy,
                    absolute_errors,
                )[0, 1]
            )
        else:
            uncertainty_error_correlation = float("nan")

        records.append(
            {
                "method": method,
                "target": target,
                "nominal_coverage": nominal_coverage,
                "empirical_coverage": empirical_coverage,
                "coverage_error": (empirical_coverage - nominal_coverage),
                "mean_interval_width": (mean_interval_width),
                "median_interval_width": (median_interval_width),
                "normalized_interval_width": (normalized_interval_width),
                "mean_absolute_error": float(np.mean(absolute_errors)),
                "uncertainty_error_correlation": (uncertainty_error_correlation),
                "sample_count": float(len(actual_values)),
            }
        )

    return pd.DataFrame(records)


def build_uncertainty_error_bins(
    actual: pd.DataFrame,
    ensemble_intervals: pd.DataFrame,
    targets: Sequence[str],
    bin_count: int = 3,
) -> pd.DataFrame:
    """Compare prediction errors across uncertainty levels."""
    if bin_count < 1:
        raise ValueError("bin_count must be at least one.")

    target_names = tuple(targets)
    records: list[dict[str, float | int | str]] = []

    for target in target_names:
        prediction_column = f"prediction_{target}"
        uncertainty_column = f"uncertainty_std_{target}"

        _require_columns(
            frame=actual,
            columns=(target,),
            frame_name="Actual values",
        )
        _require_columns(
            frame=ensemble_intervals,
            columns=(
                prediction_column,
                uncertainty_column,
            ),
            frame_name="Ensemble intervals",
        )

        actual_values = actual[target].to_numpy(dtype=np.float64)
        prediction_values = ensemble_intervals[prediction_column].to_numpy(dtype=np.float64)
        uncertainty_values = ensemble_intervals[uncertainty_column].to_numpy(dtype=np.float64)

        if len(actual_values) == 0:
            raise ValueError("At least one observation is required.")

        absolute_errors = np.abs(actual_values - prediction_values)

        sorted_indices = np.argsort(uncertainty_values)
        group_count = min(bin_count, len(sorted_indices))
        index_groups = np.array_split(
            sorted_indices,
            group_count,
        )

        for bin_index, indices in enumerate(
            index_groups,
            start=1,
        ):
            records.append(
                {
                    "target": target,
                    "uncertainty_bin": bin_index,
                    "sample_count": int(len(indices)),
                    "mean_uncertainty_std": float(np.mean(uncertainty_values[indices])),
                    "mean_absolute_error": float(np.mean(absolute_errors[indices])),
                }
            )

    return pd.DataFrame(records)


def plot_interval_calibration(
    coverage_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot empirical coverage against requested coverage."""
    _require_columns(
        frame=coverage_metrics,
        columns=(
            "method",
            "target",
            "nominal_coverage",
            "empirical_coverage",
        ),
        frame_name="Coverage metrics",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.scatter(
        coverage_metrics["nominal_coverage"],
        coverage_metrics["empirical_coverage"],
        s=55,
    )

    for row in coverage_metrics.itertuples(index=False):
        axis.annotate(
            f"{row.target}\n{row.method}",
            (
                row.nominal_coverage,
                row.empirical_coverage,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Prediction Interval Calibration")
    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def plot_uncertainty_error_bins(
    uncertainty_bins: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot error by predicted uncertainty group."""
    _require_columns(
        frame=uncertainty_bins,
        columns=(
            "target",
            "uncertainty_bin",
            "mean_absolute_error",
        ),
        frame_name="Uncertainty bins",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(9, 6))

    for target, target_frame in uncertainty_bins.groupby(
        "target",
        sort=True,
    ):
        target_frame = target_frame.sort_values("uncertainty_bin")

        axis.plot(
            target_frame["uncertainty_bin"],
            target_frame["mean_absolute_error"],
            marker="o",
            label=target,
        )

    axis.set_title("Prediction Error by Uncertainty Level")
    axis.set_xlabel("Uncertainty bin")
    axis.set_ylabel("Mean absolute error")
    axis.set_xticks(sorted(uncertainty_bins["uncertainty_bin"].unique()))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path

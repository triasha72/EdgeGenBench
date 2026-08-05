"""End-to-end uncertainty evaluation for EdgeGenBench."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from edgegenbench.evaluation.calibration import (
    build_uncertainty_error_bins,
    evaluate_prediction_intervals,
    plot_interval_calibration,
    plot_uncertainty_error_bins,
)
from edgegenbench.models.fp32_linear import DEFAULT_TARGETS
from edgegenbench.models.preprocessing import FEATURE_COLUMNS
from edgegenbench.models.tree_surrogate import (
    RANDOM_FOREST,
    TreeSurrogate,
)
from edgegenbench.uncertainty.conformal import (
    build_conformal_intervals,
    calculate_conformal_quantiles,
)
from edgegenbench.uncertainty.ensemble import (
    predict_tree_ensemble_intervals,
)

DEFAULT_COVERAGES = (
    0.80,
    0.90,
    0.95,
)


@dataclass(frozen=True)
class UncertaintyArtifacts:
    """Artifacts created by uncertainty evaluation."""

    model_path: Path
    ensemble_intervals_path: Path
    conformal_quantiles_path: Path
    conformal_interval_paths: tuple[Path, ...]
    coverage_metrics_path: Path
    uncertainty_bins_path: Path
    summary_path: Path
    plot_paths: tuple[Path, ...]
    calibration_rows: int
    test_rows: int


def _load_dataset(
    dataset_path: Path,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Load and validate an EdgeGenBench dataset."""
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

    available_splits = set(frame["split"].astype(str))
    required_splits = {
        "train",
        "validation",
        "test",
    }
    missing_splits = sorted(required_splits.difference(available_splits))

    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")

    return frame


def _load_best_parameters(
    summary_path: Path,
) -> dict[str, object]:
    """Read selected Random-Forest parameters."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Random-Forest summary does not exist: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    parameters = summary.get("best_parameters")

    if not isinstance(parameters, dict):
        raise ValueError("Random-Forest summary is missing best_parameters.")

    return parameters


def _split_training_calibration(
    training_pool: pd.DataFrame,
    calibration_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic proper-training and calibration sets."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one.")

    if len(training_pool) < 4:
        raise ValueError("At least four training observations are required.")

    random_generator = np.random.default_rng(random_state)
    shuffled_positions = random_generator.permutation(len(training_pool))

    calibration_count = max(
        1,
        int(np.ceil(len(training_pool) * calibration_fraction)),
    )

    if calibration_count >= len(training_pool):
        raise ValueError("Calibration split leaves no training data.")

    calibration_positions = shuffled_positions[:calibration_count]
    training_positions = shuffled_positions[calibration_count:]

    proper_training = training_pool.iloc[training_positions].copy()

    calibration = training_pool.iloc[calibration_positions].copy()

    return proper_training, calibration


def _attach_actual_values(
    intervals: pd.DataFrame,
    actual: pd.DataFrame,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Add actual target values to an interval report."""
    report = intervals.reset_index(drop=True).copy()

    for target_index, target in enumerate(targets):
        report.insert(
            target_index,
            f"actual_{target}",
            actual[target].reset_index(drop=True).to_numpy(),
        )

    return report


def _coverage_label(coverage: float) -> str:
    """Create a stable percentage label."""
    return f"{int(round(coverage * 100.0)):02d}"


def evaluate_uncertainty(
    dataset_path: Path,
    random_forest_summary_path: Path = Path("artifacts/tree_baselines/random_forest/summary.json"),
    output_dir: Path = Path("artifacts/uncertainty"),
    targets: Sequence[str] = DEFAULT_TARGETS,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    calibration_fraction: float = 0.20,
    ensemble_coverage: float = 0.90,
    random_state: int = 42,
) -> UncertaintyArtifacts:
    """Train a calibration model and evaluate uncertainty."""
    target_names = tuple(targets)

    if not target_names:
        raise ValueError("At least one target must be supplied.")

    coverage_values = tuple(sorted({float(value) for value in coverages}))

    if not coverage_values:
        raise ValueError("At least one conformal coverage is required.")

    for coverage in (
        *coverage_values,
        ensemble_coverage,
    ):
        if not 0.0 < coverage < 1.0:
            raise ValueError("All coverage values must be between zero and one.")

    frame = _load_dataset(
        dataset_path=dataset_path,
        targets=target_names,
    )

    best_parameters = _load_best_parameters(random_forest_summary_path)

    training_pool = frame.loc[frame["split"] == "train"].copy()

    proper_training, calibration_frame = _split_training_calibration(
        training_pool=training_pool,
        calibration_fraction=calibration_fraction,
        random_state=random_state,
    )

    test_frame = frame.loc[frame["split"] == "test"].copy()

    model = TreeSurrogate.fit(
        frame=proper_training,
        model_type=RANDOM_FOREST,
        targets=target_names,
        parameters=best_parameters,
        random_state=random_state,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / "calibration_random_forest.joblib"
    model.save(model_path)

    calibration_predictions = model.predict(calibration_frame)
    test_predictions = model.predict(test_frame)

    ensemble_intervals = predict_tree_ensemble_intervals(
        model=model,
        frame=test_frame,
        coverage=ensemble_coverage,
    )

    ensemble_intervals_path = output_dir / "ensemble_intervals.csv"

    ensemble_report = _attach_actual_values(
        intervals=ensemble_intervals,
        actual=test_frame,
        targets=target_names,
    )
    ensemble_report.to_csv(
        ensemble_intervals_path,
        index=False,
    )

    coverage_frames: list[pd.DataFrame] = []

    coverage_frames.append(
        evaluate_prediction_intervals(
            actual=test_frame,
            intervals=ensemble_intervals,
            targets=target_names,
            nominal_coverage=ensemble_coverage,
            method="random_forest_tree_quantiles",
        )
    )

    quantile_records: list[dict[str, float | str]] = []
    conformal_interval_paths: list[Path] = []

    for coverage in coverage_values:
        quantiles = calculate_conformal_quantiles(
            actual=calibration_frame,
            predicted=calibration_predictions,
            targets=target_names,
            coverage=coverage,
        )

        for target in target_names:
            quantile_records.append(
                {
                    "coverage": coverage,
                    "target": target,
                    "conformal_quantile": float(quantiles[target]),
                }
            )

        conformal_intervals = build_conformal_intervals(
            predictions=test_predictions,
            quantiles=quantiles,
            targets=target_names,
        )

        interval_path = output_dir / (f"conformal_intervals_{_coverage_label(coverage)}.csv")

        conformal_report = _attach_actual_values(
            intervals=conformal_intervals,
            actual=test_frame,
            targets=target_names,
        )
        conformal_report.to_csv(
            interval_path,
            index=False,
        )

        conformal_interval_paths.append(interval_path)

        coverage_frames.append(
            evaluate_prediction_intervals(
                actual=test_frame,
                intervals=conformal_intervals,
                targets=target_names,
                nominal_coverage=coverage,
                method="split_conformal",
            )
        )

    conformal_quantiles = pd.DataFrame(quantile_records)

    conformal_quantiles_path = output_dir / "conformal_quantiles.csv"
    conformal_quantiles.to_csv(
        conformal_quantiles_path,
        index=False,
    )

    coverage_metrics = pd.concat(
        coverage_frames,
        ignore_index=True,
    )

    coverage_metrics_path = output_dir / "coverage_metrics.csv"
    coverage_metrics.to_csv(
        coverage_metrics_path,
        index=False,
    )

    uncertainty_bins = build_uncertainty_error_bins(
        actual=test_frame,
        ensemble_intervals=ensemble_intervals,
        targets=target_names,
        bin_count=3,
    )

    uncertainty_bins_path = output_dir / "uncertainty_error_bins.csv"
    uncertainty_bins.to_csv(
        uncertainty_bins_path,
        index=False,
    )

    calibration_plot_path = output_dir / "interval_calibration.png"
    uncertainty_error_plot_path = output_dir / "uncertainty_error_bins.png"

    plot_paths = (
        plot_interval_calibration(
            coverage_metrics=coverage_metrics,
            output_path=calibration_plot_path,
        ),
        plot_uncertainty_error_bins(
            uncertainty_bins=uncertainty_bins,
            output_path=(uncertainty_error_plot_path),
        ),
    )

    summary_path = output_dir / "uncertainty_summary.json"

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "model_type": RANDOM_FOREST,
        "best_parameters": best_parameters,
        "random_state": random_state,
        "calibration_fraction": (calibration_fraction),
        "ensemble_coverage": ensemble_coverage,
        "conformal_coverages": list(coverage_values),
        "proper_training_rows": int(len(proper_training)),
        "calibration_rows": int(len(calibration_frame)),
        "test_rows": int(len(test_frame)),
        "model_path": str(model_path),
        "ensemble_intervals_path": str(ensemble_intervals_path),
        "conformal_quantiles_path": str(conformal_quantiles_path),
        "coverage_metrics_path": str(coverage_metrics_path),
        "uncertainty_bins_path": str(uncertainty_bins_path),
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

    return UncertaintyArtifacts(
        model_path=model_path,
        ensemble_intervals_path=(ensemble_intervals_path),
        conformal_quantiles_path=(conformal_quantiles_path),
        conformal_interval_paths=tuple(conformal_interval_paths),
        coverage_metrics_path=(coverage_metrics_path),
        uncertainty_bins_path=(uncertainty_bins_path),
        summary_path=summary_path,
        plot_paths=plot_paths,
        calibration_rows=len(calibration_frame),
        test_rows=len(test_frame),
    )

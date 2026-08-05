"""Command-line interface for EdgeGenBench."""

from pathlib import Path

import typer

from edgegenbench import __version__
from edgegenbench.data.generate import generate_dataset
from edgegenbench.evaluation.model_comparison import (
    compare_model_artifacts,
)
from edgegenbench.training.feasibility import (
    train_feasibility_classifier,
)
from edgegenbench.training.fp32_baseline import (
    train_fp32_baseline,
)
from edgegenbench.training.tree_baselines import (
    train_tree_baselines,
)
from edgegenbench.uncertainty.pipeline import (
    evaluate_uncertainty,
)

app = typer.Typer(
    add_completion=False,
    help="EdgeGenBench: surrogate-model and edge-inference benchmarking.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run EdgeGenBench commands."""


@app.command(name="info")
def info() -> None:
    """Show the installed project version and status."""
    typer.echo(f"EdgeGenBench {__version__}")
    typer.echo("Status: project scaffold ready.")


@app.command(name="generate-data")
def generate_data(
    config: Path = typer.Option(
        Path("configs/v0_1.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the dataset configuration file.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Optional directory that overrides the configured output location.",
    ),
) -> None:
    """Generate a reproducible synthetic aircraft-design benchmark dataset."""
    artifacts = generate_dataset(
        config_path=config,
        output_dir=output_dir,
    )

    typer.echo(f"Created dataset: {artifacts.data_path}")
    typer.echo(f"Created metadata: {artifacts.metadata_path}")
    typer.echo(f"Rows: {artifacts.row_count}")
    typer.echo(f"Feasible fraction: {artifacts.feasible_fraction:.1%}")


@app.command(name="train-fp32-baseline")
def train_fp32(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the generated EdgeGenBench CSV dataset.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/fp32_baseline"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for model and evaluation artifacts.",
    ),
) -> None:
    """Train and evaluate the FP32 linear surrogate baseline."""
    artifacts = train_fp32_baseline(
        dataset_path=dataset,
        output_dir=output_dir,
    )

    typer.echo(f"Best alpha: {artifacts.best_alpha:g}")
    typer.echo(f"Mean test NRMSE: {artifacts.mean_test_nrmse_std:.6f}")
    typer.echo(f"Mean test R2: {artifacts.mean_test_r2:.6f}")
    typer.echo(f"Saved model: {artifacts.model_path}")
    typer.echo(f"Saved summary: {artifacts.summary_path}")


@app.command(name="train-tree-baselines")
def train_trees(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the generated EdgeGenBench CSV dataset.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/tree_baselines"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for nonlinear model artifacts.",
    ),
) -> None:
    """Train and evaluate nonlinear tree-based baselines."""
    model_artifacts = train_tree_baselines(
        dataset_path=dataset,
        output_dir=output_dir,
    )

    for artifact in model_artifacts:
        typer.echo(f"Model: {artifact.model_type}")
        typer.echo(f"  Mean test NRMSE: {artifact.mean_test_nrmse_std:.6f}")
        typer.echo(f"  Mean test R2: {artifact.mean_test_r2:.6f}")
        typer.echo(f"  Saved model: {artifact.model_path}")
        typer.echo(f"  Saved summary: {artifact.summary_path}")


@app.command(name="compare-models")
def compare_models(
    artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--artifact-root",
        "-a",
        file_okay=False,
        dir_okay=True,
        help="Root directory containing model artifacts.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/model_comparison"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for comparison tables and plots.",
    ),
) -> None:
    """Compare completed surrogate-model training runs."""
    artifacts = compare_model_artifacts(
        artifact_root=artifact_root,
        output_dir=output_dir,
    )

    typer.echo("Models compared: 3")
    typer.echo(f"Best accuracy model: {artifacts.best_accuracy_model}")
    typer.echo(f"Best mean R2 model: {artifacts.best_mean_r2_model}")
    typer.echo(f"Lowest batch-1 latency: {artifacts.lowest_latency_model}")
    typer.echo(f"Smallest model: {artifacts.smallest_model}")
    typer.echo(f"Saved report: {artifacts.aggregate_metrics_path.parent}")


@app.command(name="evaluate-uncertainty")
def evaluate_uncertainty_command(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the generated benchmark dataset.",
    ),
    random_forest_summary: Path = typer.Option(
        Path("artifacts/tree_baselines/random_forest/summary.json"),
        "--random-forest-summary",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Random-Forest model-selection summary.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/uncertainty"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for uncertainty artifacts.",
    ),
    calibration_fraction: float = typer.Option(
        0.20,
        "--calibration-fraction",
        min=0.01,
        max=0.99,
        help="Fraction of training rows used for calibration.",
    ),
) -> None:
    """Evaluate ensemble and conformal uncertainty."""
    artifacts = evaluate_uncertainty(
        dataset_path=dataset,
        random_forest_summary_path=(random_forest_summary),
        output_dir=output_dir,
        calibration_fraction=calibration_fraction,
    )

    typer.echo(f"Calibration rows: {artifacts.calibration_rows}")
    typer.echo(f"Test rows: {artifacts.test_rows}")
    typer.echo(f"Coverage metrics: {artifacts.coverage_metrics_path}")
    typer.echo(f"Uncertainty summary: {artifacts.summary_path}")


@app.command(name="train-feasibility-classifier")
def train_feasibility(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Path to the generated EdgeGenBench dataset."),
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/feasibility_classifier"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help=("Directory for feasibility classifier artifacts."),
    ),
    max_false_safe_rate: float = typer.Option(
        0.05,
        "--max-false-safe-rate",
        min=0.0,
        max=1.0,
        help=("Maximum validation false-safe rate used for threshold selection."),
    ),
) -> None:
    """Train and evaluate the feasibility classifier."""
    artifacts = train_feasibility_classifier(
        dataset_path=dataset,
        output_dir=output_dir,
        max_false_safe_rate=(max_false_safe_rate),
    )

    typer.echo(f"Selected threshold: {artifacts.selected_threshold:.2f}")

    typer.echo(f"Test false-safe rate: {artifacts.false_safe_rate:.4f}")

    typer.echo(f"Test balanced accuracy: {artifacts.balanced_accuracy:.4f}")

    typer.echo(f"Test rows: {artifacts.test_rows}")

    typer.echo(f"Saved model: {artifacts.model_path}")

    typer.echo(f"Saved summary: {artifacts.summary_path}")

"""Command-line interface for EdgeGenBench."""

from pathlib import Path

import typer

from edgegenbench import __version__
from edgegenbench.data.generate import generate_dataset
from edgegenbench.training.fp32_baseline import train_fp32_baseline

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
    artifacts = generate_dataset(config_path=config, output_dir=output_dir)

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

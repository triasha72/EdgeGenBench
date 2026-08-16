"""Command-line interface for EdgeGenBench."""

from pathlib import Path

import typer

from edgegenbench import __version__
from edgegenbench.data.generate import generate_dataset
from edgegenbench.deployment.benchmark import (
    benchmark_edge_models,
)
from edgegenbench.deployment.neural_benchmark import (
    benchmark_neural_onnx,
)
from edgegenbench.deployment.neural_fp16 import (
    export_neural_surrogate_fp16,
)
from edgegenbench.deployment.neural_fp16_benchmark import (
    benchmark_neural_fp16,
)
from edgegenbench.deployment.neural_int8 import (
    export_neural_surrogate_int8,
)
from edgegenbench.deployment.neural_int8_benchmark import (
    benchmark_neural_int8,
)
from edgegenbench.deployment.neural_onnx_export import (
    export_neural_surrogate_onnx,
)
from edgegenbench.deployment.onnx_export import (
    export_edge_models,
)
from edgegenbench.evaluation.model_comparison import (
    compare_model_artifacts,
)
from edgegenbench.evaluation.physics_validation import (
    validate_optimization_designs,
)
from edgegenbench.optimization.pipeline import (
    optimize_designs,
)
from edgegenbench.training.feasibility import (
    train_feasibility_classifier,
)
from edgegenbench.training.fp32_baseline import (
    train_fp32_baseline,
)
from edgegenbench.training.neural_surrogate import (
    train_neural_surrogate,
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
    typer.echo("Status: compact neural edge-inference benchmark.")


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


@app.command(name="train-neural-surrogate")
def train_neural_surrogate_command(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Generated EdgeGenBench training dataset.",
    ),
    config: Path = typer.Option(
        Path("configs/neural_v0_2.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Neural-surrogate training configuration.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/neural_surrogate"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for neural-surrogate artifacts.",
    ),
) -> None:
    """Train and evaluate the compact PyTorch surrogate."""
    artifacts = train_neural_surrogate(
        dataset_path=dataset,
        config_path=config,
        output_dir=output_dir,
    )

    typer.echo(f"Device: {artifacts.device}")
    typer.echo(f"Parameters: {artifacts.parameter_count}")
    typer.echo(f"Best epoch: {artifacts.best_epoch}")
    typer.echo(f"Best validation loss: {artifacts.best_validation_loss:.6f}")
    typer.echo(f"Mean test NRMSE: {artifacts.mean_test_nrmse_std:.6f}")
    typer.echo(f"Mean test R2: {artifacts.mean_test_r2:.6f}")
    typer.echo(f"Saved model: {artifacts.model_path}")
    typer.echo(f"Saved summary: {artifacts.summary_path}")


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
        random_forest_summary_path=random_forest_summary,
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
        help="Path to the generated EdgeGenBench dataset.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/feasibility_classifier"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for feasibility classifier artifacts.",
    ),
    max_false_safe_rate: float = typer.Option(
        0.05,
        "--max-false-safe-rate",
        min=0.0,
        max=1.0,
        help="Maximum validation false-safe rate used for threshold selection.",
    ),
) -> None:
    """Train and evaluate the feasibility classifier."""
    artifacts = train_feasibility_classifier(
        dataset_path=dataset,
        output_dir=output_dir,
        max_false_safe_rate=max_false_safe_rate,
    )

    typer.echo(f"Selected threshold: {artifacts.selected_threshold:.2f}")
    typer.echo(f"Test false-safe rate: {artifacts.false_safe_rate:.4f}")
    typer.echo(f"Test balanced accuracy: {artifacts.balanced_accuracy:.4f}")
    typer.echo(f"Test rows: {artifacts.test_rows}")
    typer.echo(f"Saved model: {artifacts.model_path}")
    typer.echo(f"Saved summary: {artifacts.summary_path}")


@app.command(name="optimize-designs")
def optimize_designs_command(
    config: Path = typer.Option(
        Path("configs/optimization_v0_1.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Optimization configuration.",
    ),
    surrogate_model: Path = typer.Option(
        Path("artifacts/tree_baselines/random_forest/model.joblib"),
        "--surrogate-model",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Trained surrogate model.",
    ),
    feasibility_model: Path = typer.Option(
        Path("artifacts/feasibility_classifier/model.joblib"),
        "--feasibility-model",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Trained feasibility classifier.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/optimization"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for optimization artifacts.",
    ),
) -> None:
    """Run constrained multi-objective optimization."""
    artifacts = optimize_designs(
        config_path=config,
        surrogate_model_path=surrogate_model,
        feasibility_model_path=feasibility_model,
        output_dir=output_dir,
    )

    typer.echo(f"Candidates: {artifacts.candidate_count}")
    typer.echo(f"Feasible candidates: {artifacts.feasible_count}")
    typer.echo(f"Feasible fraction: {artifacts.feasible_fraction:.2%}")
    typer.echo(f"Pareto designs: {artifacts.pareto_count}")
    typer.echo(f"Safety threshold: {artifacts.feasibility_threshold:.2f}")
    typer.echo(f"Saved summary: {artifacts.summary_path}")


@app.command(name="validate-optimization")
def validate_optimization_command(
    designs: Path = typer.Option(
        Path("artifacts/optimization/representative_designs.csv"),
        "--designs",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Optimized designs to validate.",
    ),
    benchmark_config: Path = typer.Option(
        Path("configs/v0_1.yaml"),
        "--benchmark-config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Physics benchmark configuration.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/optimization_validation"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for validation artifacts.",
    ),
) -> None:
    """Validate optimized designs against the physics model."""
    artifacts = validate_optimization_designs(
        designs_path=designs,
        benchmark_config_path=benchmark_config,
        output_dir=output_dir,
    )

    typer.echo(f"Designs validated: {artifacts.design_count}")
    typer.echo(f"Targets validated: {artifacts.validated_target_count}")
    typer.echo(f"Feasibility agreement: {artifacts.feasibility_agreement_rate:.2%}")
    typer.echo(f"Saved metrics: {artifacts.metrics_path}")
    typer.echo(f"Saved summary: {artifacts.summary_path}")


@app.command(name="export-edge-models")
def export_edge_models_command(
    surrogate_model: Path = typer.Option(
        Path("artifacts/tree_baselines/random_forest/model.joblib"),
        "--surrogate-model",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Trained surrogate model.",
    ),
    feasibility_model: Path = typer.Option(
        Path("artifacts/feasibility_classifier/model.joblib"),
        "--feasibility-model",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Trained feasibility classifier.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/edge_export"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for exported ONNX models.",
    ),
) -> None:
    """Export trained estimators to ONNX."""
    artifacts = export_edge_models(
        surrogate_model_path=surrogate_model,
        feasibility_model_path=feasibility_model,
        output_dir=output_dir,
    )

    typer.echo(f"Encoded features: {artifacts.feature_count}")
    typer.echo(f"Surrogate targets: {artifacts.surrogate_target_count}")
    typer.echo(f"Safety threshold: {artifacts.feasibility_threshold:.2f}")
    typer.echo(f"Surrogate ONNX: {artifacts.surrogate_onnx_path}")
    typer.echo(f"Feasibility ONNX: {artifacts.feasibility_onnx_path}")
    typer.echo(f"Metadata: {artifacts.metadata_path}")


@app.command(name="benchmark-edge-models")
def benchmark_edge_models_command(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Benchmark dataset.",
    ),
    surrogate_model: Path = typer.Option(
        Path("artifacts/tree_baselines/random_forest/model.joblib"),
        "--surrogate-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    feasibility_model: Path = typer.Option(
        Path("artifacts/feasibility_classifier/model.joblib"),
        "--feasibility-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    surrogate_onnx: Path = typer.Option(
        Path("artifacts/edge_export/surrogate.onnx"),
        "--surrogate-onnx",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    feasibility_onnx: Path = typer.Option(
        Path("artifacts/edge_export/feasibility.onnx"),
        "--feasibility-onnx",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    metadata: Path = typer.Option(
        Path("artifacts/edge_export/metadata.json"),
        "--metadata",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/edge_benchmark"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Benchmark ONNX Runtime against Scikit-learn."""
    artifacts = benchmark_edge_models(
        dataset_path=dataset,
        surrogate_model_path=surrogate_model,
        feasibility_model_path=feasibility_model,
        surrogate_onnx_path=surrogate_onnx,
        feasibility_onnx_path=feasibility_onnx,
        metadata_path=metadata,
        output_dir=output_dir,
    )

    typer.echo(f"Test rows: {artifacts.test_rows}")
    typer.echo(f"Classifier agreement: {artifacts.classifier_agreement:.2%}")
    typer.echo(f"Maximum surrogate difference: {artifacts.max_surrogate_absolute_error:.6g}")
    typer.echo(f"Equivalence report: {artifacts.equivalence_path}")
    typer.echo(f"Latency report: {artifacts.latency_path}")
    typer.echo(f"Summary: {artifacts.summary_path}")


@app.command(name="export-neural-onnx")
def export_neural_onnx_command(
    model: Path = typer.Option(
        Path("artifacts/neural_surrogate/model.pt"),
        "--model",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Trained neural-surrogate checkpoint.",
    ),
    preprocessing: Path = typer.Option(
        Path("artifacts/neural_surrogate/preprocessing.npz"),
        "--preprocessing",
        "-p",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Frozen neural preprocessing artifact.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/neural_onnx"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for neural ONNX export artifacts.",
    ),
    opset: int = typer.Option(
        18,
        "--opset",
        min=1,
        help="Target ONNX opset.",
    ),
) -> None:
    """Export the compact PyTorch surrogate to ONNX."""
    artifacts = export_neural_surrogate_onnx(
        model_path=model,
        preprocessing_path=preprocessing,
        output_dir=output_dir,
        target_opset=opset,
    )

    typer.echo(f"Input dimension: {artifacts.input_dim}")
    typer.echo(f"Output dimension: {artifacts.output_dim}")
    typer.echo(f"ONNX opset: {artifacts.target_opset}")
    typer.echo(f"ONNX size: {artifacts.onnx_size_bytes} bytes")
    typer.echo(f"ONNX model: {artifacts.onnx_path}")
    typer.echo(f"Metadata: {artifacts.metadata_path}")


@app.command(name="benchmark-neural-onnx")
def benchmark_neural_onnx_command(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Benchmark dataset.",
    ),
    model: Path = typer.Option(
        Path("artifacts/neural_surrogate/model.pt"),
        "--model",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Trained neural-surrogate checkpoint.",
    ),
    preprocessing: Path = typer.Option(
        Path("artifacts/neural_surrogate/preprocessing.npz"),
        "--preprocessing",
        "-p",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Frozen neural preprocessing artifact.",
    ),
    onnx_model: Path = typer.Option(
        Path("artifacts/neural_onnx/neural_surrogate.onnx"),
        "--onnx-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Exported neural ONNX graph.",
    ),
    metadata: Path = typer.Option(
        Path("artifacts/neural_onnx/metadata.json"),
        "--metadata",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Neural ONNX export metadata.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/neural_onnx_benchmark"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for neural ONNX benchmark artifacts.",
    ),
    repeats: int = typer.Option(
        500,
        "--repeats",
        min=1,
        help="Measured latency repetitions per runtime and batch size.",
    ),
    warmups: int = typer.Option(
        50,
        "--warmups",
        min=0,
        help="Warmup iterations per runtime and batch size.",
    ),
) -> None:
    """Benchmark PyTorch CPU against ONNX Runtime CPU."""
    artifacts = benchmark_neural_onnx(
        dataset_path=dataset,
        model_path=model,
        preprocessing_path=preprocessing,
        onnx_model_path=onnx_model,
        metadata_path=metadata,
        output_dir=output_dir,
        batch_sizes=(
            1,
            32,
            256,
        ),
        repeats=repeats,
        warmups=warmups,
    )

    typer.echo(f"Test rows: {artifacts.test_rows}")
    typer.echo(f"Numerically equivalent: {artifacts.normalized_equivalent}")
    typer.echo(
        f"Normalized mean absolute difference: {artifacts.normalized_mean_absolute_difference:.10e}"
    )
    typer.echo(
        "Normalized maximum absolute difference: "
        f"{artifacts.normalized_max_absolute_difference:.10e}"
    )
    typer.echo(f"Equivalence report: {artifacts.equivalence_path}")
    typer.echo(f"Latency report: {artifacts.latency_path}")
    typer.echo(f"Summary: {artifacts.summary_path}")


@app.command(name="export-neural-fp16")
def export_neural_fp16_command(
    fp32_model: Path = typer.Option(
        Path("artifacts/neural_onnx/neural_surrogate.onnx"),
        "--fp32-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Canonical dynamic FP32 neural ONNX graph.",
    ),
    fp32_metadata: Path = typer.Option(
        Path("artifacts/neural_onnx/metadata.json"),
        "--fp32-metadata",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Metadata for the canonical FP32 neural ONNX graph.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/neural_fp16"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for FP16 neural ONNX artifacts.",
    ),
) -> None:
    """Convert the canonical neural ONNX graph from FP32 to FP16."""
    artifacts = export_neural_surrogate_fp16(
        fp32_model_path=fp32_model,
        fp32_metadata_path=fp32_metadata,
        output_dir=output_dir,
    )

    typer.echo(f"Input dimension: {artifacts.input_dim}")
    typer.echo(f"Output dimension: {artifacts.output_dim}")
    typer.echo(f"FP32 model size: {artifacts.fp32_model_size_bytes} bytes")
    typer.echo(f"FP16 model size: {artifacts.fp16_model_size_bytes} bytes")
    typer.echo(f"Model-size reduction: {artifacts.size_reduction_percent:.2f}%")
    typer.echo(f"FP16 initializer count: {artifacts.fp16_initializer_count}")
    typer.echo(f"FP16 model: {artifacts.onnx_path}")
    typer.echo(f"Metadata: {artifacts.metadata_path}")


@app.command(name="benchmark-neural-fp16")
def benchmark_neural_fp16_command(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Benchmark dataset containing the neural test split.",
    ),
    preprocessing: Path = typer.Option(
        Path("artifacts/neural_surrogate/preprocessing.npz"),
        "--preprocessing",
        "-p",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Frozen neural preprocessing artifact.",
    ),
    fp32_model: Path = typer.Option(
        Path("artifacts/neural_onnx/neural_surrogate.onnx"),
        "--fp32-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Canonical dynamic FP32 neural ONNX graph.",
    ),
    fp16_model: Path = typer.Option(
        Path("artifacts/neural_fp16/neural_surrogate_fp16.onnx"),
        "--fp16-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Converted dynamic FP16 neural ONNX graph.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/neural_fp16_benchmark"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for FP16 accuracy and CoreML benchmark artifacts.",
    ),
    runs: int = typer.Option(
        5,
        "--runs",
        min=1,
        help="Independent paired FP32/FP16 latency runs.",
    ),
    repeats: int = typer.Option(
        500,
        "--repeats",
        min=1,
        help="Measured latency repetitions per precision and batch size.",
    ),
    warmups: int = typer.Option(
        50,
        "--warmups",
        min=0,
        help="Warmup iterations per precision and batch size.",
    ),
    max_mean_normalized_drift: float = typer.Option(
        0.002,
        "--max-mean-normalized-drift",
        min=0.0,
        help="EdgeGenBench regression ceiling for mean normalized FP16 drift.",
    ),
    max_normalized_drift: float = typer.Option(
        0.012,
        "--max-normalized-drift",
        min=0.0,
        help="EdgeGenBench regression ceiling for maximum normalized FP16 drift.",
    ),
) -> None:
    """Benchmark FP32 and FP16 neural ONNX models on CoreML."""
    artifacts = benchmark_neural_fp16(
        dataset_path=dataset,
        preprocessing_path=preprocessing,
        fp32_model_path=fp32_model,
        fp16_model_path=fp16_model,
        output_dir=output_dir,
        batch_sizes=(
            1,
            32,
            256,
        ),
        runs=runs,
        repeats=repeats,
        warmups=warmups,
        max_mean_normalized_drift=max_mean_normalized_drift,
        max_normalized_drift=max_normalized_drift,
    )
    typer.echo(f"Test rows: {artifacts.test_rows}")
    typer.echo(f"Mean normalized FP16 drift: {artifacts.normalized_mean_absolute_difference:.10e}")
    typer.echo(
        f"Maximum normalized FP16 drift: {artifacts.normalized_max_absolute_difference:.10e}"
    )
    typer.echo(f"Mean drift within limit: {artifacts.mean_drift_within_limit}")
    typer.echo(f"Maximum drift within limit: {artifacts.max_drift_within_limit}")
    typer.echo(f"FP16 mean NRMSE: {artifacts.fp16_mean_nrmse_std:.10f}")
    typer.echo(f"FP16 mean R2: {artifacts.fp16_mean_r2:.10f}")
    typer.echo(f"Equivalence report: {artifacts.equivalence_path}")
    typer.echo(f"Task metrics: {artifacts.task_metrics_path}")
    typer.echo(f"Latency runs: {artifacts.latency_runs_path}")
    typer.echo(f"Latency summary: {artifacts.latency_summary_path}")
    typer.echo(f"Summary: {artifacts.summary_path}")


@app.command(name="export-neural-int8")
def export_neural_int8_command(
    fp32_model: Path = typer.Option(
        Path("artifacts/neural_onnx/neural_surrogate.onnx"),
        "--fp32-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Canonical dynamic FP32 neural ONNX graph."),
    ),
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Dataset containing the training split used for INT8 calibration."),
    ),
    preprocessing: Path = typer.Option(
        Path("artifacts/neural_surrogate/preprocessing.npz"),
        "--preprocessing",
        "-p",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Frozen neural preprocessing artifact."),
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/neural_int8"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help=("Directory for mixed-precision INT8 neural ONNX artifacts."),
    ),
) -> None:
    """Export the validated mixed-precision INT8/FP32 neural ONNX model."""
    artifacts = export_neural_surrogate_int8(
        fp32_model_path=(fp32_model),
        dataset_path=(dataset),
        preprocessing_path=(preprocessing),
        output_dir=(output_dir),
    )

    typer.echo(f"Input dimension: {artifacts.input_dim}")
    typer.echo(f"Output dimension: {artifacts.output_dim}")
    typer.echo(f"Calibration rows: {artifacts.calibration_rows}")
    typer.echo(f"FP32 model size: {artifacts.fp32_model_size_bytes} bytes")
    typer.echo(f"INT8 model size: {artifacts.int8_model_size_bytes} bytes")
    typer.echo(f"Model-size reduction: {artifacts.size_reduction_percent:.2f}%")
    typer.echo(f"INT8 initializer count: {artifacts.int8_initializer_count}")
    typer.echo(f"INT32 initializer count: {artifacts.int32_initializer_count}")
    typer.echo(f"Excluded FP32 nodes: {', '.join(artifacts.excluded_nodes)}")
    typer.echo(f"INT8 model: {artifacts.onnx_path}")
    typer.echo(f"Metadata: {artifacts.metadata_path}")


@app.command(name="benchmark-neural-int8")
def benchmark_neural_int8_command(
    dataset: Path = typer.Option(
        Path("data/raw/edgegenbench_v0_1.csv"),
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Benchmark dataset containing the neural test split."),
    ),
    preprocessing: Path = typer.Option(
        Path("artifacts/neural_surrogate/preprocessing.npz"),
        "--preprocessing",
        "-p",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Frozen neural preprocessing artifact."),
    ),
    fp32_model: Path = typer.Option(
        Path("artifacts/neural_onnx/neural_surrogate.onnx"),
        "--fp32-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Canonical dynamic FP32 neural ONNX graph."),
    ),
    int8_model: Path = typer.Option(
        Path("artifacts/neural_int8/neural_surrogate_int8.onnx"),
        "--int8-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=("Mixed-precision INT8/FP32 neural ONNX graph."),
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/neural_int8_benchmark"),
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help=("Directory for INT8 quality and CPU benchmark artifacts."),
    ),
    runs: int = typer.Option(
        5,
        "--runs",
        min=1,
        help=("Independent paired FP32/INT8 latency runs."),
    ),
    repeats: int = typer.Option(
        500,
        "--repeats",
        min=1,
        help=("Measured latency repetitions per precision and batch size."),
    ),
    warmups: int = typer.Option(
        50,
        "--warmups",
        min=0,
        help=("Warmup iterations per precision and batch size."),
    ),
    max_mean_normalized_drift: float = typer.Option(
        0.015,
        "--max-mean-normalized-drift",
        min=0.0,
        help=("Regression ceiling for mean normalized INT8 drift."),
    ),
    max_p99_normalized_drift: float = typer.Option(
        0.040,
        "--max-p99-normalized-drift",
        min=0.0,
        help=("Regression ceiling for P99 normalized INT8 drift."),
    ),
    max_p999_normalized_drift: float = typer.Option(
        0.060,
        "--max-p999-normalized-drift",
        min=0.0,
        help=("Regression ceiling for P99.9 normalized INT8 drift."),
    ),
    max_normalized_drift: float = typer.Option(
        0.080,
        "--max-normalized-drift",
        min=0.0,
        help=("Regression ceiling for maximum normalized INT8 drift."),
    ),
) -> None:
    """Benchmark FP32 and mixed-precision INT8 neural ONNX models on CPU."""
    artifacts = benchmark_neural_int8(
        dataset_path=(dataset),
        preprocessing_path=(preprocessing),
        fp32_model_path=(fp32_model),
        int8_model_path=(int8_model),
        output_dir=(output_dir),
        batch_sizes=(
            1,
            32,
            256,
        ),
        runs=runs,
        repeats=repeats,
        warmups=warmups,
        max_mean_normalized_drift=(max_mean_normalized_drift),
        max_p99_normalized_drift=(max_p99_normalized_drift),
        max_p999_normalized_drift=(max_p999_normalized_drift),
        max_normalized_drift=(max_normalized_drift),
    )

    typer.echo(f"Test rows: {artifacts.test_rows}")
    typer.echo(f"Mean normalized INT8 drift: {artifacts.normalized_mean_absolute_difference:.10e}")
    typer.echo(f"P95 normalized INT8 drift: {artifacts.normalized_p95_absolute_difference:.10e}")
    typer.echo(f"P99 normalized INT8 drift: {artifacts.normalized_p99_absolute_difference:.10e}")
    typer.echo(f"P99.9 normalized INT8 drift: {artifacts.normalized_p999_absolute_difference:.10e}")
    typer.echo(
        f"Maximum normalized INT8 drift: {artifacts.normalized_max_absolute_difference:.10e}"
    )
    typer.echo(f"Mean drift within limit: {artifacts.mean_drift_within_limit}")
    typer.echo(f"P99 drift within limit: {artifacts.p99_drift_within_limit}")
    typer.echo(f"P99.9 drift within limit: {artifacts.p999_drift_within_limit}")
    typer.echo(f"Maximum drift within limit: {artifacts.max_drift_within_limit}")
    typer.echo(f"INT8 mean NRMSE: {artifacts.int8_mean_nrmse_std:.10f}")
    typer.echo(f"INT8 mean R2: {artifacts.int8_mean_r2:.10f}")
    typer.echo(f"Equivalence report: {artifacts.equivalence_path}")
    typer.echo(f"Task metrics: {artifacts.task_metrics_path}")
    typer.echo(f"Latency runs: {artifacts.latency_runs_path}")
    typer.echo(f"Latency summary: {artifacts.latency_summary_path}")
    typer.echo(f"Summary: {artifacts.summary_path}")

"""CLI tests for the neural-surrogate workflow."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

import edgegenbench.cli as cli_module
from edgegenbench.cli import app

runner = CliRunner()

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(
    text: str,
) -> str:
    """Remove ANSI terminal escape sequences from captured CLI output."""
    return _ANSI_ESCAPE_RE.sub(
        "",
        text,
    )


def test_neural_training_command_is_registered() -> None:
    """Verify that neural training is exposed through the CLI."""
    result = runner.invoke(
        app,
        [
            "train-neural-surrogate",
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "Train and evaluate the compact PyTorch surrogate." in output

    assert "--dataset" in output
    assert "--config" in output
    assert "--output-dir" in output


def test_neural_onnx_export_command_is_registered() -> None:
    """Verify that neural ONNX export is exposed through the CLI."""
    result = runner.invoke(
        app,
        [
            "export-neural-onnx",
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "Export the compact PyTorch surrogate to ONNX." in output

    assert "--model" in output
    assert "--preprocessing" in output
    assert "--output-dir" in output
    assert "--opset" in output


def test_neural_onnx_benchmark_command_is_registered() -> None:
    """Verify that neural ONNX benchmarking is exposed through the CLI."""
    result = runner.invoke(
        app,
        [
            "benchmark-neural-onnx",
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "Benchmark PyTorch CPU against ONNX Runtime CPU." in output

    assert "--dataset" in output
    assert "--model" in output
    assert "--preprocessing" in output
    assert "--onnx-model" in output
    assert "--metadata" in output
    assert "--repeats" in output
    assert "--warmups" in output


def test_neural_fp16_export_command_is_registered() -> None:
    """Verify that FP16 neural ONNX export is exposed through the CLI."""
    result = runner.invoke(
        app,
        [
            "export-neural-fp16",
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "Convert the canonical neural ONNX graph from FP32 to FP16." in output

    assert "--fp32-model" in output
    assert "--fp32-metadata" in output
    assert "--output-dir" in output


def test_neural_fp16_benchmark_command_is_registered() -> None:
    """Verify that FP16 CoreML benchmarking is exposed through the CLI."""
    result = runner.invoke(
        app,
        [
            "benchmark-neural-fp16",
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "Benchmark FP32 and FP16 neural ONNX models on CoreML." in output

    assert "--dataset" in output
    assert "--preprocessing" in output
    assert "--fp32-model" in output
    assert "--fp16-model" in output
    assert "--output-dir" in output
    assert "--runs" in output
    assert "--repeats" in output
    assert "--warmups" in output


def test_neural_fp16_benchmark_long_options_are_parsed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that the complete FP16 drift option names are accepted."""
    dataset_path = tmp_path / "dataset.csv"

    preprocessing_path = tmp_path / "preprocessing.npz"

    fp32_model_path = tmp_path / "fp32.onnx"

    fp16_model_path = tmp_path / "fp16.onnx"

    output_dir = tmp_path / "benchmark"

    dataset_path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    preprocessing_path.touch()
    fp32_model_path.touch()
    fp16_model_path.touch()

    received: dict[
        str,
        Any,
    ] = {}

    def fake_benchmark_neural_fp16(
        **kwargs: Any,
    ) -> SimpleNamespace:
        received.update(kwargs)

        return SimpleNamespace(
            test_rows=10,
            normalized_mean_absolute_difference=(0.001),
            normalized_max_absolute_difference=(0.010),
            mean_drift_within_limit=True,
            max_drift_within_limit=True,
            fp16_mean_nrmse_std=0.05,
            fp16_mean_r2=0.99,
            equivalence_path=(output_dir / "equivalence.csv"),
            task_metrics_path=(output_dir / "task_metrics.csv"),
            latency_runs_path=(output_dir / "latency_runs.csv"),
            latency_summary_path=(output_dir / "latency_summary.csv"),
            summary_path=(output_dir / "summary.json"),
        )

    monkeypatch.setattr(
        cli_module,
        "benchmark_neural_fp16",
        fake_benchmark_neural_fp16,
    )

    result = runner.invoke(
        app,
        [
            "benchmark-neural-fp16",
            "--dataset",
            str(dataset_path),
            "--preprocessing",
            str(preprocessing_path),
            "--fp32-model",
            str(fp32_model_path),
            "--fp16-model",
            str(fp16_model_path),
            "--output-dir",
            str(output_dir),
            "--runs",
            "2",
            "--repeats",
            "7",
            "--warmups",
            "3",
            "--max-mean-normalized-drift",
            "0.003",
            "--max-normalized-drift",
            "0.015",
        ],
    )

    assert result.exit_code == 0

    assert received["dataset_path"] == dataset_path

    assert received["preprocessing_path"] == preprocessing_path

    assert received["fp32_model_path"] == fp32_model_path

    assert received["fp16_model_path"] == fp16_model_path

    assert received["output_dir"] == output_dir

    assert received["batch_sizes"] == (
        1,
        32,
        256,
    )

    assert received["runs"] == 2

    assert received["repeats"] == 7

    assert received["warmups"] == 3

    assert received["max_mean_normalized_drift"] == 0.003

    assert received["max_normalized_drift"] == 0.015

    output = _strip_ansi(result.stdout)

    assert "Test rows: 10" in output

    assert "Mean drift within limit: True" in output

    assert "Maximum drift within limit: True" in output


def test_root_help_lists_fp16_commands() -> None:
    """Verify that both FP16 commands appear in root CLI help."""
    result = runner.invoke(
        app,
        [
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "export-neural-fp16" in output

    assert "benchmark-neural-fp16" in output


def test_neural_int8_export_command_is_registered() -> None:
    """Verify that mixed-precision INT8 export is exposed through the CLI."""
    result = runner.invoke(
        app,
        [
            "export-neural-int8",
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "Export the validated mixed-precision INT8/FP32 neural ONNX model." in output

    assert "--fp32-model" in output

    assert "--dataset" in output

    assert "--preprocessing" in output

    assert "--output-dir" in output


def test_neural_int8_benchmark_command_is_registered() -> None:
    """Verify that INT8 CPU benchmarking is exposed through the CLI."""
    result = runner.invoke(
        app,
        [
            "benchmark-neural-int8",
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "Benchmark FP32 and mixed-precision INT8 neural ONNX models on CPU." in output

    assert "--dataset" in output

    assert "--preprocessing" in output

    assert "--fp32-model" in output

    assert "--int8-model" in output

    assert "--output-dir" in output

    assert "--runs" in output

    assert "--repeats" in output

    assert "--warmups" in output


def test_neural_int8_benchmark_long_options_are_parsed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify complete INT8 regression option names are accepted."""
    dataset_path = tmp_path / "dataset.csv"

    preprocessing_path = tmp_path / "preprocessing.npz"

    fp32_model_path = tmp_path / "fp32.onnx"

    int8_model_path = tmp_path / "int8.onnx"

    output_dir = tmp_path / "benchmark"

    dataset_path.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    preprocessing_path.touch()
    fp32_model_path.touch()
    int8_model_path.touch()

    received: dict[
        str,
        Any,
    ] = {}

    def fake_benchmark_neural_int8(
        **kwargs: Any,
    ) -> SimpleNamespace:
        received.update(kwargs)

        return SimpleNamespace(
            test_rows=10,
            normalized_mean_absolute_difference=0.008,
            normalized_p95_absolute_difference=0.020,
            normalized_p99_absolute_difference=0.028,
            normalized_p999_absolute_difference=0.042,
            normalized_max_absolute_difference=0.060,
            mean_drift_within_limit=True,
            p99_drift_within_limit=True,
            p999_drift_within_limit=True,
            max_drift_within_limit=True,
            int8_mean_nrmse_std=0.052,
            int8_mean_r2=0.996,
            equivalence_path=(output_dir / "equivalence.csv"),
            task_metrics_path=(output_dir / "task_metrics.csv"),
            latency_runs_path=(output_dir / "latency_runs.csv"),
            latency_summary_path=(output_dir / "latency_summary.csv"),
            summary_path=(output_dir / "summary.json"),
        )

    monkeypatch.setattr(
        cli_module,
        "benchmark_neural_int8",
        fake_benchmark_neural_int8,
    )

    result = runner.invoke(
        app,
        [
            "benchmark-neural-int8",
            "--dataset",
            str(dataset_path),
            "--preprocessing",
            str(preprocessing_path),
            "--fp32-model",
            str(fp32_model_path),
            "--int8-model",
            str(int8_model_path),
            "--output-dir",
            str(output_dir),
            "--runs",
            "2",
            "--repeats",
            "7",
            "--warmups",
            "3",
            "--max-mean-normalized-drift",
            "0.016",
            "--max-p99-normalized-drift",
            "0.041",
            "--max-p999-normalized-drift",
            "0.061",
            "--max-normalized-drift",
            "0.081",
        ],
    )

    assert result.exit_code == 0

    assert received["dataset_path"] == dataset_path

    assert received["preprocessing_path"] == preprocessing_path

    assert received["fp32_model_path"] == fp32_model_path

    assert received["int8_model_path"] == int8_model_path

    assert received["output_dir"] == output_dir

    assert received["batch_sizes"] == (
        1,
        32,
        256,
    )

    assert received["runs"] == 2

    assert received["repeats"] == 7

    assert received["warmups"] == 3

    assert received["max_mean_normalized_drift"] == 0.016

    assert received["max_p99_normalized_drift"] == 0.041

    assert received["max_p999_normalized_drift"] == 0.061

    assert received["max_normalized_drift"] == 0.081

    output = _strip_ansi(result.stdout)

    assert "Test rows: 10" in output

    assert "Mean drift within limit: True" in output

    assert "P99 drift within limit: True" in output

    assert "P99.9 drift within limit: True" in output

    assert "Maximum drift within limit: True" in output


def test_root_help_lists_int8_commands() -> None:
    """Verify that both INT8 commands appear in root CLI help."""
    result = runner.invoke(
        app,
        [
            "--help",
        ],
    )

    assert result.exit_code == 0

    output = _strip_ansi(result.stdout)

    assert "export-neural-int8" in output

    assert "benchmark-neural-int8" in output

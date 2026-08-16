"""CLI tests for the neural-surrogate workflow."""

import re

from typer.testing import CliRunner

from edgegenbench.cli import app

runner = CliRunner()

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
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

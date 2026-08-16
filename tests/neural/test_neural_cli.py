"""CLI tests for the neural-surrogate workflow."""

import re

from typer.testing import CliRunner

from edgegenbench.cli import app

runner = CliRunner()

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI terminal escape sequences from captured CLI output."""
    return _ANSI_ESCAPE_RE.sub("", text)


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

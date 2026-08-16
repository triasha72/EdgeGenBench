"""CLI tests for the neural-surrogate workflow."""

from typer.testing import CliRunner

from edgegenbench.cli import app

runner = CliRunner()


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
    assert "Train and evaluate the compact PyTorch surrogate." in result.stdout
    assert "--dataset" in result.stdout
    assert "--config" in result.stdout
    assert "--output-dir" in result.stdout

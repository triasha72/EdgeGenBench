"""Smoke tests for the EdgeGenBench command-line interface."""

from typer.testing import CliRunner

from edgegenbench import __version__
from edgegenbench.cli import app


def test_info_command() -> None:
    """Verify that the info command reports the installed release."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["info"],
    )

    assert result.exit_code == 0
    assert f"EdgeGenBench {__version__}" in result.stdout
    assert "Status: compact neural edge-inference benchmark." in result.stdout

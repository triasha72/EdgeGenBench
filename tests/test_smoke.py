from typer.testing import CliRunner

from edgegenbench.cli import app


def test_info_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "EdgeGenBench 0.1.0" in result.stdout

"""Tests for Typer CLI."""

from typer.testing import CliRunner
from memora.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Memora version" in result.stdout


def test_cli_hardware():
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Memora Hardware Discovery" in result.stdout

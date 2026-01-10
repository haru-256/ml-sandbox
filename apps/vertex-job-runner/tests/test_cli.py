import re

from pytest_console_scripts import ScriptRunner

PRG = "vrun"


def test_help(script_runner: ScriptRunner) -> None:
    result = script_runner.run([PRG, "--help"], check=True)
    assert re.search(rf"Usage: {PRG}", result.stdout) is not None


def test_dry_run_config_loading(script_runner: ScriptRunner) -> None:
    result = script_runner.run([PRG, "--dry-run"], check=True)
    # Check if command is correctly split into a list in the output table
    assert "['uv', 'run', 'main.py']" in result.stdout
    assert "['hoge=2', 'fuga=1']" in result.stdout
    assert "haru256-vertex-ai-sandbox" in result.stdout


def test_cli_args_override(script_runner: ScriptRunner) -> None:
    result = script_runner.run([PRG, "--args", "param1=val1 param2=val2", "--dry-run"], check=True)
    assert "['param1=val1', 'param2=val2']" in result.stdout

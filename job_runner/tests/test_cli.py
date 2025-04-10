import re

from pytest_console_scripts import ScriptRunner

PRG = "job-runner"


def test_help(script_runner: ScriptRunner) -> None:
    result = script_runner.run([PRG, "--help"], check=True)
    assert re.match(rf"usage: {PRG}", result.stdout) is not None

#!/usr/bin/env python3
"""Read a pinned CLI version from a mise.toml [tools] table."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path


def fail(message: str) -> None:
    """Print an error to stderr and exit with status 1.

    Args:
        message: Human-readable failure reason.

    Raises:
        SystemExit: Always raised with code 1.
    """
    print(message, file=sys.stderr)
    raise SystemExit(1)


def read_tool_version(file_path: Path, tool: str) -> str:
    """Return the string pin for ``[tools].<tool>`` in a mise.toml file.

    Args:
        file_path: Absolute path to the mise TOML file.
        tool: Key under the ``[tools]`` table, such as ``uv`` or ``terraform``.

    Returns:
        The stripped version string.

    Raises:
        SystemExit: If the file is missing, is not valid TOML, or the tool pin
            is missing, empty, or not a string.
    """
    if not tool:
        fail("tool input is empty")
    if not file_path.is_file():
        fail(f"mise file not found: {file_path}")

    try:
        with file_path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        fail(f"failed to parse {file_path}: {exc}")

    tools = data.get("tools")
    if not isinstance(tools, dict):
        fail(f"[tools] table is missing in {file_path}")
    if tool not in tools:
        fail(f"[tools].{tool} is missing in {file_path}")

    version = tools[tool]
    if not isinstance(version, str) or not version.strip():
        fail(f"[tools].{tool} must be a non-empty string in {file_path}")
    return version.strip()


def main() -> None:
    """Read env inputs and write ``version=<pin>`` to ``GITHUB_OUTPUT``.

    Raises:
        SystemExit: If required environment variables are missing or the mise
            file cannot be read.
    """
    tool = os.environ.get("MISE_TOOL", "")
    file_path = Path(os.environ.get("MISE_FILE", ""))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        fail("GITHUB_OUTPUT is not set")

    version = read_tool_version(file_path, tool)
    with open(github_output, "a", encoding="utf-8") as fh:
        fh.write(f"version={version}\n")


if __name__ == "__main__":
    main()

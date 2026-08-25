#!/usr/bin/env python3
"""Write the mise.toml [tools] pin for MISE_TOOL to GITHUB_OUTPUT."""

import os
import sys
import tomllib
from pathlib import Path

tool = os.environ.get("MISE_TOOL", "").strip()
workspace = os.environ.get("GITHUB_WORKSPACE", "")
path = Path(workspace) / "mise.toml"
github_output = os.environ.get("GITHUB_OUTPUT")

if not tool:
    sys.exit("tool input is empty")
if not github_output:
    sys.exit("GITHUB_OUTPUT is not set")
if not path.is_file():
    sys.exit(f"mise file not found: {path}")

with path.open("rb") as fh:
    data = tomllib.load(fh)

tools = data.get("tools")
if not isinstance(tools, dict) or tool not in tools:
    sys.exit(f"[tools].{tool} is missing in {path}")
version = tools[tool]
if not isinstance(version, str) or not version.strip():
    sys.exit(f"[tools].{tool} must be a non-empty string in {path}")

with open(github_output, "a", encoding="utf-8") as fh:
    fh.write(f"version={version.strip()}\n")

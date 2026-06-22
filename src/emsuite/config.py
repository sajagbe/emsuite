"""Shared configuration file parsing for EMSuite input files."""

from __future__ import annotations

import ast
from pathlib import Path


def parse_assignments(content: str) -> dict:
    """
    Parse ``key = value`` assignments from a config file body.

    Uses ``ast.literal_eval`` for values (no arbitrary code execution).
    Lines starting with ``#`` and blank lines are ignored.
    """
    params: dict = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


def parse_config_file(filepath: str | Path, defaults: dict | None = None) -> dict:
    """
    Parse a config file and optionally merge with defaults.

    When *defaults* is provided, only keys present in *defaults* are returned
    (surface-style parsing). When omitted, all assignments are returned
    (tuning-style parsing).
    """
    path = Path(filepath)
    if not path.exists():
        return defaults.copy() if defaults is not None else {}

    content = path.read_text()
    parsed = parse_assignments(content)

    if defaults is None:
        return parsed

    params = defaults.copy()
    for key in defaults:
        if key in parsed:
            params[key] = parsed[key]
    return params

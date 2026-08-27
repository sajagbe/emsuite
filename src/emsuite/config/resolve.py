"""Unified config resolution: merge built-in defaults < file/dict < explicit kwargs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parser import parse_config_file


class _Unset:
    """Sentinel for an omitted keyword argument (renders cleanly in signatures)."""

    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


# Sentinel marking a keyword argument the caller did not pass. Lets the kwargs
# API distinguish "user omitted this" (defer to config/defaults) from an
# explicit value, so a config file and a couple of overrides can be mixed.
UNSET: Any = _Unset()


def resolve_config(
    config: str | Path | dict | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve channel parameters from up to three layers.

    Precedence (lowest to highest):

    1. ``defaults`` — built-in channel defaults
    2. ``config`` — a path to a ``.in`` file or a dict of parameters
    3. ``overrides`` — explicit keyword arguments (values equal to :data:`UNSET`
       are ignored, so unspecified kwargs never clobber the layers below)

    Returns a plain dict; validation is left to the channel.
    """
    params: dict[str, Any] = dict(defaults or {})

    if config is not None:
        if isinstance(config, dict):
            loaded = dict(config)
        elif isinstance(config, (str, Path)):
            loaded = parse_config_file(config)
        else:
            raise TypeError(f"config must be a path, dict, or None; got {type(config).__name__}")
        params.update(loaded)

    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not UNSET})

    return params

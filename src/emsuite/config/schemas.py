"""Per-channel config validation helpers."""

from __future__ import annotations

from typing import Any


class ConfigValidationError(ValueError):
    """Raised when a channel input file is missing required keys."""


def require_keys(params: dict[str, Any], required: list[str], channel: str) -> None:
    missing = [key for key in required if params.get(key) in (None, "")]
    if missing:
        raise ConfigValidationError(
            f"{channel} input missing required parameter(s): {', '.join(missing)}"
        )


def validate_surface_params(params: dict[str, Any]) -> dict[str, Any]:
    require_keys(params, ["input_type", "input_data"], "surface")
    return params


def validate_tuning_params(params: dict[str, Any]) -> dict[str, Any]:
    molecule = params.get("molecule") or params.get("xyz_file")
    if not molecule:
        raise ConfigValidationError("tuning input missing required parameter: molecule")
    require_keys(params, ["surface_file"], "tuning")
    params["molecule"] = molecule
    return params


def validate_potential_params(params: dict[str, Any]) -> dict[str, Any]:
    require_keys(params, ["molecule"], "potential")
    return params


def validate_coupled_params(params: dict[str, Any]) -> dict[str, Any]:
    require_keys(params, ["molecule"], "coupled")
    if not params.get("properties"):
        raise ConfigValidationError("coupled input missing required parameter: properties")
    return params

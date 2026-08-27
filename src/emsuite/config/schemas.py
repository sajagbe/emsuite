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


_POTENTIAL_METHODS = frozenset({"apbs", "coulomb"})
_FUTURE_POTENTIAL_METHODS = frozenset({"esp", "mep"})
_POTENTIAL_QUANTITIES = frozenset({"potential", "charge"})


def validate_potential_params(params: dict[str, Any]) -> dict[str, Any]:
    require_keys(params, ["molecule"], "potential")
    method = str(params.get("method") or "apbs").lower()
    quantity = str(params.get("quantity") or "potential").lower()
    if method in _FUTURE_POTENTIAL_METHODS:
        raise ConfigValidationError(
            f"method={method!r} is not implemented yet (planned PySCF ESP/MEP backend)"
        )
    if method not in _POTENTIAL_METHODS:
        allowed = ", ".join(sorted(_POTENTIAL_METHODS | _FUTURE_POTENTIAL_METHODS))
        raise ConfigValidationError(f"potential method must be one of: {allowed}")
    if quantity not in _POTENTIAL_QUANTITIES:
        raise ConfigValidationError("potential quantity must be 'potential' or 'charge'")
    if quantity == "charge" and method != "apbs":
        raise ConfigValidationError(
            "quantity='charge' requires method='apbs' (Gauss-law needs dielectric maps)"
        )
    params["method"] = method
    params["quantity"] = quantity
    return params


def validate_coupled_params(params: dict[str, Any]) -> dict[str, Any]:
    require_keys(params, ["molecule"], "coupled")
    if not params.get("properties"):
        raise ConfigValidationError("coupled input missing required parameter: properties")
    return params

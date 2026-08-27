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


_CORE_PROPERTIES = frozenset(
    {
        "gse",
        "homo",
        "lumo",
        "gap",
        "dm",
        "spin",
        "ie",
        "ea",
        "cp",
        "eng",
        "hard",
        "efl",
        "nfl",
        "fukui_plus",
        "fukui_minus",
        "exe",
        "osc",
        "all",
    }
)


def validate_tuning_params(params: dict[str, Any]) -> dict[str, Any]:
    molecule = params.get("molecule") or params.get("xyz_file")
    if not molecule:
        raise ConfigValidationError("tuning input missing required parameter: molecule")
    require_keys(params, ["surface_file"], "tuning")
    params["molecule"] = molecule
    properties = params.get("properties")
    if properties:
        if isinstance(properties, str):
            properties = [properties]
        unknown = [p for p in properties if p not in _CORE_PROPERTIES]
        if unknown:
            raise ConfigValidationError(
                f"unknown tuning property: {unknown[0]!r} "
                f"(removed or not supported; core set is {sorted(_CORE_PROPERTIES - {'all'})})"
            )
    return params


_POTENTIAL_METHODS = frozenset({"apbs"})
_FUTURE_POTENTIAL_METHODS = frozenset({"esp", "mep"})
_POTENTIAL_QUANTITIES = frozenset({"potential", "charge"})
_REMOVED_BOND_SCAN_KEYS = ("bond_scan_atoms", "bond_scan_steps", "bond_scan_span")


def validate_potential_params(params: dict[str, Any]) -> dict[str, Any]:
    ligand = params.get("ligand") or params.get("molecule")
    if not ligand:
        raise ConfigValidationError(
            "potential input missing required parameter: molecule or ligand"
        )
    params["molecule"] = ligand
    params["ligand"] = ligand
    method = str(params.get("method") or "apbs").lower()
    quantity = str(params.get("quantity") or "potential").lower()
    ligand_atoms = str(params.get("ligand_atoms") or "present").lower()
    if method == "coulomb":
        raise ConfigValidationError(
            "method='coulomb' has been removed; use method='apbs'"
        )
    if method in _FUTURE_POTENTIAL_METHODS:
        raise ConfigValidationError(
            f"method={method!r} is not implemented yet (planned PySCF ESP/MEP backend)"
        )
    if method not in _POTENTIAL_METHODS:
        allowed = ", ".join(sorted(_POTENTIAL_METHODS | _FUTURE_POTENTIAL_METHODS))
        raise ConfigValidationError(f"potential method must be one of: {allowed}")
    if quantity not in _POTENTIAL_QUANTITIES:
        raise ConfigValidationError("potential quantity must be 'potential' or 'charge'")
    if ligand_atoms not in {"present", "absent"}:
        raise ConfigValidationError("ligand_atoms must be 'present' or 'absent'")
    if ligand_atoms == "absent" and not params.get("protein"):
        raise ConfigValidationError("ligand_atoms='absent' requires protein=")
    for key in _REMOVED_BOND_SCAN_KEYS:
        if params.get(key) not in (None, "", False):
            raise ConfigValidationError(
                f"{key} has been removed (bond-axis scan is no longer supported)"
            )
    params["method"] = method
    params["quantity"] = quantity
    params["ligand_atoms"] = ligand_atoms
    return params


def validate_coupled_params(params: dict[str, Any]) -> dict[str, Any]:
    ligand = params.get("ligand") or params.get("molecule")
    if not ligand:
        raise ConfigValidationError("coupled input missing required parameter: molecule or ligand")
    params["molecule"] = ligand
    params["ligand"] = ligand
    if not params.get("properties"):
        raise ConfigValidationError("coupled input missing required parameter: properties")
    potential_keys = {
        "molecule": ligand,
        "ligand": ligand,
        "protein": params.get("protein"),
        "ligand_atoms": params.get("ligand_atoms"),
        "method": params.get("potential_method"),
        "quantity": params.get("potential_quantity"),
    }
    validate_potential_params({k: v for k, v in potential_keys.items() if v is not None})
    properties = params.get("properties")
    if properties:
        validate_tuning_params(
            {
                "molecule": ligand,
                "surface_file": params.get("surface_file") or "_",
                "properties": properties,
            }
        )
    return params

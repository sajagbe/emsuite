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
    import math

    for key, allowed in (
        ("input_type", {"XYZ", "SMILES"}),
        ("surface_type", {"homogenous", "heterogenous"}),
    ):
        if key not in params:
            continue
        value = str(params[key])
        value = value.upper() if key == "input_type" else value.lower()
        if value not in allowed:
            raise ConfigValidationError(f"{key} must be one of {sorted(allowed)}")
        params[key] = value
    for key in ("surface_density", "surface_scale", "surface_charge"):
        if key not in params:
            continue
        try:
            value = float(params[key])
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(f"{key} must be a finite number") from exc
        if not math.isfinite(value) or (key != "surface_charge" and value <= 0):
            raise ConfigValidationError(
                f"{key} must be finite" + (" and positive" if key != "surface_charge" else "")
            )
        params[key] = value
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
        "exe",
        "osc",
        "all",
    }
)


def validate_tuning_params(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params.get("resume", True), bool):
        raise ConfigValidationError("resume must be True or False")
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


_REMOVED_BOND_SCAN_KEYS = ("bond_scan_atoms", "bond_scan_steps", "bond_scan_span")


_REMOVED_COUPLED_KEYS: dict[str, str] = {
    "protein": "use pdb=",
    "protein_format": "only PDB input is supported; use pdb=",
    "ligand_atoms": "use off= (charge 0, radius kept) or removed= (charge 0, radius 0)",
    "ligand_resname": "use off= or removed= with the residue name",
    "ligand_chain": "use off= or removed= with RES:CHAIN:NUMBER",
    "ligand_resseq": "use off= or removed= with RES:CHAIN:NUMBER",
    "ligand_mol2": "unknown residues get RESP charges; use unknown_charge=",
}


def reject_removed_coupled_keys(params: dict[str, Any]) -> None:
    for key, hint in _REMOVED_COUPLED_KEYS.items():
        if key in params:
            raise ConfigValidationError(f"{key!r} is no longer a coupled input key; {hint}")


def validate_coupled_params(params: dict[str, Any]) -> dict[str, Any]:
    ligand = params.get("ligand") or params.get("molecule")
    if not ligand:
        raise ConfigValidationError("coupled input missing required parameter: molecule or ligand")
    params["molecule"] = ligand
    params["ligand"] = ligand
    if not params.get("properties"):
        raise ConfigValidationError("coupled input missing required parameter: properties")
    quantity = str(params.get("potential_quantity", "charge")).lower()
    if quantity != "charge":
        raise ConfigValidationError(
            "coupled potential_quantity must be 'charge': tuning requires charges in electrons, not potentials"
        )
    params["potential_quantity"] = quantity
    if params.get("surface_file") and params.get("potential_surf"):
        raise ConfigValidationError("give surface_file or potential_surf, not both")
    from pathlib import Path

    output = Path(params.get("output_surf", "coupled.surf"))
    if output.is_absolute() or ".." in output.parts:
        raise ConfigValidationError(
            "output_surf must be a relative path inside the coupled output directory"
        )
    surface = validate_surface_params(
        dict(
            input_type="XYZ",
            input_data=ligand,
            surface_density=params.get("surface_density", 0.5),
            surface_scale=params.get("surface_scale", 1.0),
        )
    )
    params.update({name: surface[name] for name in ("surface_density", "surface_scale")})
    if not params.get("potential_surf"):
        from emsuite.potential.spec import PotentialSpec

        require_keys(params, ["pdb"], "coupled")
        kwargs = potential_spec_kwargs(params)
        kwargs["surface_file"] = params.get("surface_file") or "surface.surf"
        spec = PotentialSpec(**kwargs)
        for name in ("forcefield", "off", "removed", "unknown_charge", "unknown_spin"):
            params[name] = getattr(spec, name)
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


_SHARED_POTENTIAL_KEYS = (
    "pdb",
    "surface_file",
    "output_surf",
    "forcefield",
    "add_hydrogens",
    "ph",
    "unknown_charge",
    "unknown_spin",
    "unknown_method",
    "unknown_basis",
    "resp_copies",
    "off",
    "removed",
    "pdie",
    "sdie",
    "grid_spacing",
)


def potential_spec_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Coupled-input fields mapped onto the potential channel's input."""
    kwargs = {
        key: params[key]
        for key in _SHARED_POTENTIAL_KEYS
        if params.get(key) is not None or (key == "grid_spacing" and key in params)
    }
    for coupled_key, key in (("potential_method", "method"), ("potential_quantity", "quantity")):
        if params.get(coupled_key) is not None:
            kwargs[key] = params[coupled_key]
    return kwargs

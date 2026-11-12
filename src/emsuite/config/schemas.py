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
    "surface_density": "make the surface in the surface channel; give surface_file=",
    "surface_scale": "make the surface in the surface channel; give surface_file=",
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
    if not params.get("potential_surf"):
        from emsuite.potential.spec import PotentialSpec

        require_keys(params, ["pdb", "surface_file"], "coupled")
        spec = PotentialSpec(**potential_spec_kwargs(params))
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
    kwargs = {key: params[key] for key in _SHARED_POTENTIAL_KEYS if params.get(key) is not None}
    for coupled_key, key in (("potential_method", "method"), ("potential_quantity", "quantity")):
        if params.get(coupled_key) is not None:
            kwargs[key] = params[coupled_key]
    return kwargs

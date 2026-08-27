"""Keyword-argument API for the four EMSuite channels.

This is the most accessible way to drive EMSuite from Python: call a channel
function with only the parameters you care about and let defaults handle the
rest. Every function also accepts ``config=`` (a ``.in`` file path or a dict)
so existing file-based workflows keep working, and explicit keyword arguments
always override values loaded from ``config``.

Examples
--------
>>> from emsuite import api
>>> api.surface(input_type="SMILES", input_data="CCO")
>>> api.tune(molecule="CCO.xyz", surface_file="CCO.surf",
...          properties=["homo", "gap", "stark_gap"])
>>> api.potential(molecule="CCO.xyz", quantity="potential")
>>> api.coupled(molecule="CCO.xyz", properties=["homo", "lumo"])

Mix a file with overrides:

>>> api.tune(config="tuning.in", parallel=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emsuite.config import UNSET, resolve_config
from emsuite.coupled.runner import run_coupled_calculation
from emsuite.potential.runner import run_potential_calculation
from emsuite.surface.runner import run_surface_calculation
from emsuite.tuning.runner import main as _run_tuning

__all__ = ["surface", "tune", "potential", "coupled"]

Config = str | Path | dict | None


def _overrides(local_vars: dict[str, Any]) -> dict[str, Any]:
    """Build an overrides dict from a wrapper's locals, dropping UNSET/config."""
    return {k: v for k, v in local_vars.items() if k != "config" and v is not UNSET}


def surface(
    input_type: Any = UNSET,
    input_data: Any = UNSET,
    *,
    output_surf: Any = UNSET,
    optimized_xyz: Any = UNSET,
    surface_density: Any = UNSET,
    surface_scale: Any = UNSET,
    surface_type: Any = UNSET,
    surface_charge: Any = UNSET,
    optimize: Any = UNSET,
    optimize_method: Any = UNSET,
    method: Any = UNSET,
    basis_set: Any = UNSET,
    functional: Any = UNSET,
    solvent: Any = UNSET,
    charge: Any = UNSET,
    spin: Any = UNSET,
    config: Config = None,
) -> str:
    """Generate a VDW surface from SMILES or XYZ. Returns the ``.surf`` path."""
    params = resolve_config(config, _overrides(locals()))
    return run_surface_calculation(params)


def tune(
    molecule: Any = UNSET,
    surface_file: Any = UNSET,
    *,
    properties: Any = UNSET,
    basis_set: Any = UNSET,
    method: Any = UNSET,
    functional: Any = UNSET,
    charge: Any = UNSET,
    spin: Any = UNSET,
    solvent: Any = UNSET,
    calc_type: Any = UNSET,
    state_of_interest: Any = UNSET,
    triplet: Any = UNSET,
    parallel: Any = UNSET,
    num_procs: Any = UNSET,
    config: Config = None,
):
    """Run an electrostatic tuning map. See ``PROPERTY_CONFIG`` for properties."""
    params = resolve_config(config, _overrides(locals()))
    return _run_tuning(params)


def potential(
    molecule: Any = UNSET,
    surface_file: Any = UNSET,
    *,
    output_surf: Any = UNSET,
    surface_density: Any = UNSET,
    surface_scale: Any = UNSET,
    method: Any = UNSET,
    quantity: Any = UNSET,
    pdie: Any = UNSET,
    sdie: Any = UNSET,
    charge: Any = UNSET,
    spin: Any = UNSET,
    bond_scan_atoms: Any = UNSET,
    bond_scan_steps: Any = UNSET,
    bond_scan_span: Any = UNSET,
    config: Config = None,
) -> str:
    """Map APBS potential or Gauss-law charge onto a surface. Returns the ``.surf`` path."""
    params = resolve_config(config, _overrides(locals()))
    return run_potential_calculation(params)


def coupled(
    molecule: Any = UNSET,
    surface_file: Any = UNSET,
    *,
    properties: Any = UNSET,
    output_surf: Any = UNSET,
    surface_density: Any = UNSET,
    surface_scale: Any = UNSET,
    potential_method: Any = UNSET,
    potential_quantity: Any = UNSET,
    pdie: Any = UNSET,
    sdie: Any = UNSET,
    basis_set: Any = UNSET,
    method: Any = UNSET,
    functional: Any = UNSET,
    charge: Any = UNSET,
    spin: Any = UNSET,
    solvent: Any = UNSET,
    calc_type: Any = UNSET,
    parallel: Any = UNSET,
    state_of_interest: Any = UNSET,
    triplet: Any = UNSET,
    num_procs: Any = UNSET,
    config: Config = None,
) -> None:
    """Run the potential → tuning coupled pipeline."""
    params = resolve_config(config, _overrides(locals()))
    return run_coupled_calculation(params)

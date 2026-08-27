"""Coupled potential → tuning pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from emsuite.config import parse_assignments, parse_config_file
from emsuite.config.schemas import validate_coupled_params
from emsuite.potential import run_potential_calculation
from emsuite.tuning import main as run_tuning

COUPLED_DEFAULTS = {
    "molecule": None,
    "surface_file": None,
    "output_surf": "coupled.surf",
    "surface_density": 0.5,
    "surface_scale": 1.0,
    "potential_method": "apbs",
    "potential_quantity": "charge",
    "pdie": 2.0,
    "sdie": 78.54,
    "properties": ["homo", "lumo", "gap"],
    "basis_set": "6-31G*",
    "method": "dft",
    "functional": "b3lyp",
    "charge": 0,
    "spin": 0,
    "solvent": None,
    "calc_type": "separate",
    "parallel": False,
    "state_of_interest": 2,
    "triplet": False,
    "num_procs": None,
}


def parse_coupled_input(input_file: str) -> dict:
    params = parse_config_file(input_file, defaults=COUPLED_DEFAULTS)
    parsed = parse_assignments(Path(input_file).read_text())
    for key in ("potential_method", "potential_quantity", "properties", "parallel"):
        if key in parsed:
            params[key] = parsed[key]
    return validate_coupled_params(params)


def _write_potential_input(params: dict, path: Path) -> None:
    method = str(params.get("potential_method") or "apbs")
    quantity = params.get("potential_quantity")
    if not quantity:
        quantity = "charge" if method.lower() == "apbs" else "potential"
    path.write_text(
        "\n".join(
            [
                f"molecule = {params['molecule']!r}",
                f"surface_file = {params.get('surface_file')!r}"
                if params.get("surface_file")
                else "surface_file = None",
                f"output_surf = {params['output_surf']!r}",
                f"surface_density = {params['surface_density']}",
                f"surface_scale = {params['surface_scale']}",
                f"method = {method!r}",
                f"quantity = {quantity!r}",
                f"pdie = {params['pdie']}",
                f"sdie = {params['sdie']}",
            ]
        )
        + "\n"
    )


def _write_tuning_input(params: dict, surface_file: str, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"molecule = {params['molecule']!r}",
                f"surface_file = {surface_file!r}",
                f"properties = {params['properties']!r}",
                f"basis_set = {params['basis_set']!r}",
                f"method = {params['method']!r}",
                f"functional = {params['functional']!r}",
                f"charge = {params['charge']}",
                f"spin = {params['spin']}",
                f"solvent = {params['solvent']!r}",
                f"calc_type = {params['calc_type']!r}",
                f"parallel = {params['parallel']}",
                f"state_of_interest = {params['state_of_interest']}",
                f"triplet = {params['triplet']}",
            ]
        )
        + "\n"
    )


def run_coupled_calculation(config) -> None:
    """
    Run APBS-derived surface values (potential or Gauss-law charge), then tuning maps.

    Distinct from tuning ``calc_type='combined'`` (all probes at once).

    Args:
        config (str | Path | dict): Path to a coupled.in file, or a parameter dict.
    """
    print("\n" + "=" * 60)
    print("           Coupled Potential → Tuning Pipeline")
    print("=" * 60 + "\n")

    if isinstance(config, dict):
        params = validate_coupled_params({**COUPLED_DEFAULTS, **config})
    else:
        params = parse_coupled_input(config)
    if not os.path.exists(params["molecule"]):
        raise FileNotFoundError(f"Molecule XYZ not found: {params['molecule']}")

    potential_in = Path("coupled_potential.in")
    tuning_in = Path("coupled_tuning.in")
    _write_potential_input(params, potential_in)
    surface_path = run_potential_calculation(str(potential_in))
    _write_tuning_input(params, surface_path, tuning_in)
    run_tuning(str(tuning_in))

    print("\nCoupled calculation complete.")
    print("=" * 60 + "\n")

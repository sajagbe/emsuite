"""Coupled potential → tuning pipeline."""

from __future__ import annotations

from pathlib import Path

from emsuite.config import parse_assignments, parse_config_file
from emsuite.config.schemas import validate_coupled_params
from emsuite.results import CoupledResult

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
    "ligand": None,
    "protein": None,
    "ligand_atoms": "present",
}


def parse_coupled_input(input_file: str) -> dict:
    params = parse_config_file(input_file, defaults=COUPLED_DEFAULTS)
    return validate_coupled_params(params)


def run_coupled_calculation(config) -> CoupledResult:
    """
    Run APBS-derived surface values (potential or Gauss-law charge), then tuning maps.

    Distinct from tuning ``calc_type='combined'`` (all probes at once).

    Args:
        config (str | Path | dict): Path to a coupled.in file, or a parameter dict.

    Returns:
        CoupledResult: The composed potential + tuning result.
    """
    print("\n" + "=" * 60)
    print("           Coupled Potential → Tuning Pipeline")
    print("=" * 60 + "\n")

    from emsuite.inputs import CoupledInput

    result = CoupledInput.from_any(config).run()

    print("\nCoupled calculation complete.")
    print("=" * 60 + "\n")

    return result

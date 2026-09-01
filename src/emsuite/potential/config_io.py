"""Potential channel input parsing."""

from __future__ import annotations

from emsuite.config import parse_config_file
from emsuite.config.schemas import validate_potential_params

POTENTIAL_DEFAULTS = {
    "molecule": None,
    "surface_file": None,
    "output_surf": "potential.surf",
    "surface_density": 0.5,
    "surface_scale": 1.0,
    "method": "apbs",
    "quantity": "potential",
    "pdie": 2.0,
    "sdie": 78.54,
    "charge": 0,
    "spin": 0,
    "ligand": None,
    "protein": None,
    "ligand_atoms": "present",
    "protein_format": "xyz",
    "ligand_resname": None,
    "ligand_chain": None,
    "ligand_resseq": None,
    "ligand_mol2": None,
    "forcefield": "AMBER",
    "ph": 7.0,
}


def parse_potential_input(input_file: str) -> dict:
    params = parse_config_file(input_file, defaults=POTENTIAL_DEFAULTS)
    return validate_potential_params(params)

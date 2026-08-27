"""Potential channel input parsing."""

from __future__ import annotations

from pathlib import Path

from emsuite.config import parse_assignments, parse_config_file
from emsuite.config.schemas import validate_potential_params

POTENTIAL_DEFAULTS = {
    "molecule": None,
    "surface_file": None,
    "output_surf": "potential.surf",
    "surface_density": 0.5,
    "surface_scale": 1.0,
    "method": "coulomb",
    "pdie": 2.0,
    "sdie": 78.54,
    "charge": 0,
    "spin": 0,
    "bond_scan_atoms": None,
    "bond_scan_steps": 10,
    "bond_scan_span": 3.0,
}


def parse_potential_input(input_file: str) -> dict:
    params = parse_config_file(input_file, defaults=POTENTIAL_DEFAULTS)
    parsed = parse_assignments(Path(input_file).read_text())
    if "method" in parsed:
        params["method"] = parsed["method"]
    return validate_potential_params(params)

"""Electrostatic potential maps on surfaces."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from emsuite.config.schemas import validate_potential_params
from emsuite.surface import load_surf, save_surf
from emsuite.surface.bond_scan import bond_scan_coords
from emsuite.surface.generate import generate_surface

from .apbs import run_apbs_potential
from .config_io import POTENTIAL_DEFAULTS, parse_potential_input
from .coulomb import coulomb_potential_at_points, partial_charges_from_xyz
from .pqr import read_xyz


def run_potential_calculation(config) -> str:
    """
    Compute electrostatic potential at surface points and write heterogeneous .surf.

    Uses Coulomb/Gasteiger charges by default; set method='apbs' for Poisson-Boltzmann.

    Args:
        config (str | Path | dict): Path to a potential.in file, or a parameter dict.
    """
    print("\n" + "=" * 60)
    print("              Electrostatic Potential Module")
    print("=" * 60 + "\n")

    if isinstance(config, dict):
        params = validate_potential_params({**POTENTIAL_DEFAULTS, **config})
    else:
        params = parse_potential_input(config)
    molecule = params["molecule"]
    if not os.path.exists(molecule):
        raise FileNotFoundError(f"Molecule XYZ not found: {molecule}")

    surface_file = params.get("surface_file")
    bond_atoms = params.get("bond_scan_atoms")
    if bond_atoms and len(bond_atoms) == 2:
        atom_coords = np.array([[x, y, z] for _, x, y, z in read_xyz(molecule)])
        coords = bond_scan_coords(
            atom_coords,
            int(bond_atoms[0]),
            int(bond_atoms[1]),
            n_steps=int(params.get("bond_scan_steps", 10)),
            span_angstrom=float(params.get("bond_scan_span", 3.0)),
        )
        print(f"Bond-axis scan: {len(coords)} points between atoms {bond_atoms}")
    elif surface_file and os.path.exists(surface_file):
        coords, _ = load_surf(surface_file)
        print(f"Loaded existing surface: {surface_file} ({len(coords)} points)")
    else:
        print("Generating VDW surface for potential mapping...")
        surf_path = generate_surface(
            input_type="XYZ",
            input_data=molecule,
            output_surf="_potential_tmp.surf",
            surface_density=params["surface_density"],
            surface_scale=params["surface_scale"],
            surface_type="homogenous",
            surface_charge=0.0,
            optimize=False,
        )
        coords, _ = load_surf(surf_path)
        if os.path.exists("_potential_tmp.surf"):
            os.remove("_potential_tmp.surf")

    method = str(params.get("method", "coulomb")).lower()
    atoms, charges = partial_charges_from_xyz(molecule)

    if method == "apbs":
        try:
            potentials = run_apbs_potential(
                molecule,
                coords,
                charges=charges,
                pdie=float(params["pdie"]),
                sdie=float(params["sdie"]),
            )
            print("Computed potentials with APBS")
        except Exception as exc:
            print(f"APBS failed ({exc}); falling back to Coulomb potential")
            potentials = coulomb_potential_at_points(atoms, charges, coords)
    else:
        potentials = coulomb_potential_at_points(atoms, charges, coords)
        print("Computed potentials with Coulomb/Gasteiger model")

    output_surf = params["output_surf"]
    save_surf(coords, potentials, output_surf, heterogenous=True)

    csv_path = Path(output_surf).with_suffix(".csv")
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(len(coords)), coords, potentials]),
        header="point_index,x,y,z,potential_hartree_per_e",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.8f"],
    )
    print(f"Potential map CSV: {csv_path}")
    print("=" * 60)
    return output_surf

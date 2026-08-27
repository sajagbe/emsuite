"""Electrostatic potential maps on surfaces."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from emsuite.config.schemas import validate_potential_params
from emsuite.surface import load_surf, save_surf
from emsuite.surface.bond_scan import bond_scan_coords
from emsuite.surface.generate import generate_surface

from .apbs import run_apbs_grids
from .config_io import POTENTIAL_DEFAULTS, parse_potential_input
from .coulomb import coulomb_potential_at_points, partial_charges_from_xyz
from .gauss import charges_at_points, potential_at_points
from .pqr import read_xyz

# Planned PySCF backends (ESP/MEP) — not implemented yet.
_FUTURE_METHODS = frozenset({"esp", "mep"})


def _surface_coords(params: dict) -> np.ndarray:
    molecule = params["molecule"]
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
        return coords
    if surface_file and os.path.exists(surface_file):
        coords, _ = load_surf(surface_file)
        print(f"Loaded existing surface: {surface_file} ({len(coords)} points)")
        return coords
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
    return coords


def _coulomb_values(atoms, charges, coords: np.ndarray) -> np.ndarray:
    values = coulomb_potential_at_points(atoms, charges, coords)
    print("Computed potentials with Coulomb/Gasteiger model")
    return values


def _apbs_values(
    molecule: str, coords: np.ndarray, charges, pdie: float, sdie: float, quantity: str
) -> np.ndarray:
    grids = run_apbs_grids(molecule, charges=charges, pdie=pdie, sdie=sdie)
    if quantity == "charge":
        values = charges_at_points(grids.potential, grids.dielx, grids.diely, grids.dielz, coords)
        print("Computed Gauss-law surface charges from APBS potential and dielectric maps")
        return values
    values = potential_at_points(grids.potential, coords)
    print("Interpolated APBS potentials at surface points")
    return values


def run_potential_calculation(config) -> str:
    """
    Map APBS electrostatics onto a surface and write a heterogeneous ``.surf``.

    ``quantity='potential'`` writes interpolated APBS φ at each surface point.
    ``quantity='charge'`` writes Gauss-law charges from φ and the dielectric maps.
    ``method='coulomb'`` is a vacuum 1/r fallback (potential only).

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

    coords = _surface_coords(params)
    method = str(params["method"]).lower()
    quantity = str(params["quantity"]).lower()
    atoms, charges = partial_charges_from_xyz(molecule)

    if method in _FUTURE_METHODS:
        raise NotImplementedError(
            f"method={method!r} is not implemented yet (planned PySCF ESP/MEP backend)"
        )

    if method == "apbs":
        try:
            values = _apbs_values(
                molecule,
                coords,
                charges,
                pdie=float(params["pdie"]),
                sdie=float(params["sdie"]),
                quantity=quantity,
            )
        except Exception as exc:
            if quantity == "charge":
                raise RuntimeError(
                    "APBS Gauss-law charges require potential and dielectric maps; "
                    "refusing to fall back to Coulomb"
                ) from exc
            print(f"APBS failed ({exc}); falling back to Coulomb potential")
            values = _coulomb_values(atoms, charges, coords)
    else:
        values = _coulomb_values(atoms, charges, coords)

    output_surf = params["output_surf"]
    save_surf(coords, values, output_surf, heterogenous=True)

    csv_path = Path(output_surf).with_suffix(".csv")
    value_header = "charge_e" if quantity == "charge" else "potential"
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(len(coords)), coords, values]),
        header=f"point_index,x,y,z,{value_header}",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.8f"],
    )
    print(f"Potential map CSV: {csv_path}")
    print("=" * 60)
    return output_surf

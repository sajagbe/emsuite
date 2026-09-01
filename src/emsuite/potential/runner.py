"""Electrostatic potential maps on surfaces."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from emsuite.surface import load_surf, save_surf
from emsuite.surface.generate import generate_surface

from .apbs import run_apbs_grids
from .gauss import charges_at_points, potential_at_points
from .occupancy import occupancy_atoms_and_charges

# Planned PySCF backends (ESP/MEP) — not implemented yet.
_FUTURE_METHODS = frozenset({"esp", "mep"})


def _surface_coords(params: dict) -> np.ndarray:
    molecule = params["molecule"]
    surface_file = params.get("surface_file")
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


def _apbs_values(
    coords: np.ndarray,
    atoms,
    charges,
    box_coords: np.ndarray,
    pdie: float,
    sdie: float,
    quantity: str,
    pqr_path: str | None = None,
) -> np.ndarray:
    if pqr_path is not None:
        grids = run_apbs_grids(pqr_path=pqr_path, box_coords=box_coords, pdie=pdie, sdie=sdie)
    else:
        grids = run_apbs_grids(
            atoms=atoms, charges=charges, box_coords=box_coords, pdie=pdie, sdie=sdie
        )
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

    Args:
        config (str | Path | dict): Path to a potential.in file, or a parameter dict.
    """
    print("\n" + "=" * 60)
    print("              Electrostatic Potential Module")
    print("=" * 60 + "\n")

    from emsuite.inputs import PotentialInput

    params = PotentialInput.from_any(config).to_dict()
    molecule = params["molecule"]
    if not os.path.exists(molecule):
        raise FileNotFoundError(f"Molecule XYZ not found: {molecule}")
    protein = params.get("protein")
    protein_format = str(params.get("protein_format") or "xyz")
    if protein and not os.path.exists(protein):
        label = "PDB" if protein_format == "pdb" else "XYZ"
        raise FileNotFoundError(f"Protein {label} not found: {protein}")

    coords = _surface_coords(params)
    method = str(params["method"]).lower()
    quantity = str(params["quantity"]).lower()
    atoms, charges, box_coords, pqr_path = occupancy_atoms_and_charges(
        ligand_xyz=molecule,
        protein_xyz=protein,
        ligand_atoms=str(params.get("ligand_atoms") or "present"),
        ligand_charge=int(params.get("charge") or 0),
        protein_format=protein_format,
        ligand_resname=params.get("ligand_resname"),
        ligand_chain=params.get("ligand_chain"),
        ligand_resseq=params.get("ligand_resseq"),
        ligand_mol2=params.get("ligand_mol2"),
        forcefield=str(params.get("forcefield") or "AMBER"),
        ph=params.get("ph"),
    )

    if method in _FUTURE_METHODS:
        raise NotImplementedError(
            f"method={method!r} is not implemented yet (planned PySCF ESP/MEP backend)"
        )

    try:
        values = _apbs_values(
            coords,
            atoms,
            charges,
            box_coords,
            pdie=float(params["pdie"]),
            sdie=float(params["sdie"]),
            quantity=quantity,
            pqr_path=pqr_path,
        )
    finally:
        if pqr_path is not None:
            # Clean up occupancy's temp dir (isolated/stripped PDB + PQR files)
            # from the pdb2pqr path — it has to outlive occupancy_atoms_and_charges
            # so run_apbs_grids can read it, so nothing else cleans it up.
            shutil.rmtree(Path(pqr_path).parent, ignore_errors=True)

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

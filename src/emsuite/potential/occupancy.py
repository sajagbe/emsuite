"""Protein/ligand PQR occupancy for APBS."""

from __future__ import annotations

import numpy as np

from emsuite.geometry import Geometry

from .charges import partial_charges_from_xyz


def assemble_pqr(
    ligand: Geometry,
    protein: Geometry,
    protein_charges: np.ndarray,
    ligand_atoms: str,
) -> tuple[list[tuple[str, float, float, float]], np.ndarray, np.ndarray]:
    """Build PQR atoms/charges and the APBS box coordinates.

    ``ligand_atoms='present'``: ligand atoms included at charge 0.
    ``ligand_atoms='absent'``: ligand atoms omitted from the PQR.
    Box always spans protein and ligand coordinates (KTD4).
    """
    mode = str(ligand_atoms).lower()
    if mode not in {"present", "absent"}:
        raise ValueError("ligand_atoms must be 'present' or 'absent'")
    if len(protein_charges) != len(protein.symbols):
        raise ValueError("protein_charges length does not match protein atoms")

    protein_atoms = [
        (symbol, float(x), float(y), float(z))
        for symbol, (x, y, z) in zip(protein.symbols, protein.coords, strict=True)
    ]
    box = np.vstack([protein.coords, ligand.coords])
    if mode == "absent":
        return protein_atoms, np.asarray(protein_charges, dtype=float), box

    ligand_atoms_xyz = [
        (symbol, float(x), float(y), float(z))
        for symbol, (x, y, z) in zip(ligand.symbols, ligand.coords, strict=True)
    ]
    charges = np.concatenate(
        [np.asarray(protein_charges, dtype=float), np.zeros(len(ligand.symbols))]
    )
    return protein_atoms + ligand_atoms_xyz, charges, box


def occupancy_atoms_and_charges(
    ligand_xyz: str,
    protein_xyz: str | None,
    ligand_atoms: str = "present",
    ligand_charge: int = 0,
) -> tuple[list[tuple[str, float, float, float]], np.ndarray, np.ndarray]:
    """Return (pqr_atoms, pqr_charges, box_coords) for an APBS run."""
    ligand = Geometry.from_xyz(ligand_xyz)
    if not protein_xyz:
        atoms, charges = partial_charges_from_xyz(ligand_xyz, charge=ligand_charge)
        return atoms, charges, ligand.coords

    protein = Geometry.from_xyz(protein_xyz)
    _protein_atoms, protein_charges = partial_charges_from_xyz(protein_xyz)
    return assemble_pqr(ligand, protein, protein_charges, ligand_atoms)

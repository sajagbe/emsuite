"""Coulomb electrostatic potential at surface points."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .pqr import read_xyz


def partial_charges_from_xyz(
    xyz_path: str,
) -> tuple[list[tuple[str, float, float, float]], np.ndarray]:
    atoms = read_xyz(xyz_path)
    if not atoms:
        raise ValueError(f"No atoms found in {xyz_path}")

    mol = Chem.MolFromXYZFile(xyz_path)
    if mol is None:
        raise ValueError(f"RDKit could not parse XYZ file: {xyz_path}")
    Chem.SanitizeMol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([float(atom.GetDoubleProp("_GasteigerCharge")) for atom in mol.GetAtoms()])
    if len(charges) != len(atoms):
        raise ValueError("Atom count mismatch between XYZ and RDKit molecule")
    return atoms, charges


def coulomb_potential_at_points(
    atoms: list[tuple[str, float, float, float]],
    charges: np.ndarray,
    surface_coords: np.ndarray,
) -> np.ndarray:
    """Return electrostatic potential (Hartree/e) at each surface point."""
    atom_coords = np.array([[x, y, z] for _, x, y, z in atoms])
    potentials = np.zeros(surface_coords.shape[0])
    for i, point in enumerate(surface_coords):
        distances = np.linalg.norm(atom_coords - point, axis=1)
        distances = np.maximum(distances, 1e-3)
        potentials[i] = np.sum(charges / distances)
    return potentials

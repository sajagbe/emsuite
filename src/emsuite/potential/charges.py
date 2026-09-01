"""Gasteiger partial charges for APBS PQR writing."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from emsuite.geometry import read_xyz


def partial_charges_from_xyz(
    xyz_path: str,
    charge: int = 0,
) -> tuple[list[tuple[str, float, float, float]], np.ndarray]:
    atoms = read_xyz(xyz_path)
    if not atoms:
        raise ValueError(f"No atoms found in {xyz_path}")

    mol = Chem.MolFromXYZFile(xyz_path)
    if mol is None:
        raise ValueError(f"RDKit could not parse XYZ file: {xyz_path}")
    # MolFromXYZFile has no bonds; Gasteiger charges need connectivity to propagate.
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    Chem.SanitizeMol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([float(atom.GetDoubleProp("_GasteigerCharge")) for atom in mol.GetAtoms()])
    if len(charges) != len(atoms):
        raise ValueError("Atom count mismatch between XYZ and RDKit molecule")
    return atoms, charges

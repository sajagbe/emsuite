"""Protein/ligand PQR occupancy for APBS."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from emsuite.geometry import Geometry

from .charges import partial_charges_from_xyz
from .pdb2pqr_runner import run_pdb2pqr
from .pdb_select import isolate_residue, strip_residue
from .pqr import read_pqr_coords, zero_ligand_charges


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


def _occupancy_from_pdb(
    ligand: Geometry,
    protein_pdb: str,
    ligand_atoms: str,
    ligand_resname: str,
    ligand_chain: str | None,
    ligand_resseq: int | None,
    ligand_mol2: str | None,
    forcefield: str,
    ph: float | None,
) -> tuple[None, None, np.ndarray, str]:
    """pdb2pqr-based occupancy. Returns (None, None, box_coords, pqr_path).

    ``'absent'``: strip the target residue before conversion (not in the PQR).
    ``'present'``/``'charged'``: isolate protein + only the target residue
    (drops any *other* HETATM so pdb2pqr's --ligand atom-name matching, which
    checks every HETATM residue, can't collide with them), convert with
    --ligand for its real radius, then zero its charge column for 'present'
    (charge stays real, pdb2pqr's own PEOE, for 'charged').
    """
    mode = str(ligand_atoms).lower()
    if mode not in {"present", "absent", "charged"}:
        raise ValueError("ligand_atoms must be 'present', 'absent', or 'charged'")

    work = Path(tempfile.mkdtemp(prefix="emsuite_pdb2pqr_"))
    pqr_path = work / "occupancy.pqr"

    if mode == "absent":
        stripped = strip_residue(
            protein_pdb, ligand_resname, ligand_chain, ligand_resseq, work / "stripped.pdb"
        )
        run_pdb2pqr(stripped, pqr_path, forcefield=forcefield, ligand_mol2=None, ph=ph)
    else:
        if not ligand_mol2:
            raise ValueError(f"ligand_atoms={mode!r} with protein_format='pdb' requires ligand_mol2")
        isolated = isolate_residue(
            protein_pdb, ligand_resname, ligand_chain, ligand_resseq, work / "isolated.pdb"
        )
        run_pdb2pqr(isolated, pqr_path, forcefield=forcefield, ligand_mol2=ligand_mol2, ph=ph)
        if mode == "present":
            zeroed = work / "zeroed.pqr"
            zero_ligand_charges(pqr_path, ligand_resname, ligand_resseq, zeroed)
            pqr_path = zeroed

    box = np.vstack([read_pqr_coords(pqr_path), ligand.coords])
    return None, None, box, str(pqr_path)


def occupancy_atoms_and_charges(
    ligand_xyz: str,
    protein_xyz: str | None,
    ligand_atoms: str = "present",
    ligand_charge: int = 0,
    protein_format: str = "xyz",
    ligand_resname: str | None = None,
    ligand_chain: str | None = None,
    ligand_resseq: int | None = None,
    ligand_mol2: str | None = None,
    forcefield: str = "AMBER",
    ph: float | None = 7.0,
) -> tuple[list[tuple[str, float, float, float]] | None, np.ndarray | None, np.ndarray, str | None]:
    """Return (pqr_atoms, pqr_charges, box_coords, pqr_path) for an APBS run.

    ``pqr_atoms``/``pqr_charges`` are populated for the XYZ+Gasteiger path
    (``protein_format='xyz'``, the existing behavior); ``pqr_path`` is
    populated instead for the pdb2pqr path (``protein_format='pdb'``), and the
    caller (potential/runner.py) passes whichever is set into run_apbs_grids.
    """
    ligand = Geometry.from_xyz(ligand_xyz)

    if protein_format == "pdb":
        if not protein_xyz:
            raise ValueError("protein_format='pdb' requires protein")
        if not ligand_resname:
            raise ValueError("protein_format='pdb' requires ligand_resname")
        return _occupancy_from_pdb(
            ligand,
            protein_xyz,
            ligand_atoms,
            ligand_resname,
            ligand_chain,
            ligand_resseq,
            ligand_mol2,
            forcefield,
            ph,
        )

    if not protein_xyz:
        atoms, charges = partial_charges_from_xyz(ligand_xyz, charge=ligand_charge)
        return atoms, charges, ligand.coords, None

    protein = Geometry.from_xyz(protein_xyz)
    _protein_atoms, protein_charges = partial_charges_from_xyz(protein_xyz)
    atoms, charges, box = assemble_pqr(ligand, protein, protein_charges, ligand_atoms)
    return atoms, charges, box, None

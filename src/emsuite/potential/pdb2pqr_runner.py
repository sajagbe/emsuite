"""PDB to PQR conversion via the pdb2pqr binary."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_pdb2pqr(
    pdb_path: str | Path,
    output_pqr: str | Path,
    forcefield: str = "AMBER",
    ligand_mol2: str | Path | None = None,
    ph: float | None = 7.0,
) -> Path:
    """Convert a PDB to a PQR via pdb2pqr, in whitespace-delimited column format.

    ``ligand_mol2``, if given, is passed as pdb2pqr's ``--ligand`` (its own PEOE
    charge assignment for that residue's atoms, matched by atom name against the
    PDB — see potential/pdb_select.py for why the PDB should already be isolated
    to protein + at most one HETATM residue before calling this).

    ``ph``, if not None, enables propka-based titration at that pH.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    args = ["pdb2pqr", "--ff", forcefield, "--whitespace"]
    if ligand_mol2 is not None:
        args += ["--ligand", str(ligand_mol2)]
    if ph is not None:
        args += ["--titration-state-method", "propka", "--with-ph", str(ph)]
    args += [str(pdb_path), str(output_pqr)]

    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"pdb2pqr failed (exit {result.returncode}): {result.stderr[-2000:]}")
    return Path(output_pqr)

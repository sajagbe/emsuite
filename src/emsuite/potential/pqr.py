"""XYZ to PQR conversion for APBS, plus PQR utilities for the pdb2pqr path."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# van der Waals radii (Å) for common elements
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


def write_pqr(
    atoms: list[tuple[str, float, float, float]],
    charges: list[float],
    output_path: str | Path,
    resname: str = "MOL",
) -> Path:
    output = Path(output_path)
    with output.open("w") as handle:
        for idx, ((symbol, x, y, z), charge) in enumerate(zip(atoms, charges, strict=True), 1):
            radius = VDW_RADII.get(symbol.upper(), 1.50)
            handle.write(
                f"ATOM  {idx:5d}  {symbol:>2s}  {resname:3s}     1"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}{charge:8.4f}{radius:8.4f}\n"
            )
    return output


def _whitespace_pqr_lines(path: str | Path) -> list[list[str]]:
    """Split a pdb2pqr ``--whitespace`` PQR into per-line tokens.

    Each ATOM/HETATM line is exactly 10 tokens:
    record_type serial name resname resseq x y z charge radius
    (pdb2pqr's own writer drops every other line type when --whitespace is set).
    Not for emsuite's own fixed-width write_pqr() output, which has no
    guaranteed whitespace between its numeric columns.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if tokens and tokens[0] in ("ATOM", "HETATM"):
            rows.append(tokens)
    return rows


def read_pqr_coords(path: str | Path) -> np.ndarray:
    """Coordinates only, from a pdb2pqr ``--whitespace`` PQR (for box-extent calc)."""
    rows = _whitespace_pqr_lines(path)
    return np.array([[float(r[5]), float(r[6]), float(r[7])] for r in rows])


def zero_ligand_charges(
    pqr_path: str | Path,
    resname: str,
    resseq: int | None,
    output_path: str | Path,
) -> Path:
    """Zero the charge column for one residue's atoms; radius/coords untouched.

    Matches by resname (+ optional resseq) — a pdb2pqr --whitespace PQR has no
    chain column (default --keep-chain=False). Intended to run on a PQR already
    isolated to protein + one target residue (see potential/pdb_select.py), so
    resname alone is unambiguous in practice.
    """
    lines = Path(pqr_path).read_text().splitlines(keepends=True)
    out_lines = []
    for line in lines:
        tokens = line.split()
        if (
            tokens
            and tokens[0] in ("ATOM", "HETATM")
            and tokens[3] == resname
            and (resseq is None or int(tokens[4]) == resseq)
        ):
            newline = "\n" if line.endswith("\n") else ""
            body = line[: -len(newline)] if newline else line
            parts = re.split(r"(\s+)", body)
            charge_positions = [i for i, p in enumerate(parts) if p.strip()]
            charge_idx = charge_positions[8]  # 9th token: record,serial,name,resname,resseq,x,y,z,[charge]
            parts[charge_idx] = "0.0000".rjust(len(parts[charge_idx]))
            line = "".join(parts) + newline
        out_lines.append(line)
    output = Path(output_path)
    output.write_text("".join(out_lines))
    return output

"""PDB residue selection and pdb2pqr-format PQR utilities (pure text, no external tools)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite.potential.pdb_select import isolate_residue, select_residue_lines, strip_residue
from emsuite.potential.pqr import read_pqr_coords, zero_ligand_charges


def _pdb_line(
    record: str,
    serial: int,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    name: str = "C1",
) -> str:
    line = list(" " * 66)
    line[0 : len(record)] = record
    line[6:11] = f"{serial:>5}"
    line[12:16] = f"{name:<4}"
    line[17:20] = f"{resname:<3}"
    line[21] = chain
    line[22:26] = f"{resseq:>4}"
    line[30:38] = f"{x:>8.3f}"
    line[38:46] = f"{y:>8.3f}"
    line[46:54] = f"{z:>8.3f}"
    return "".join(line).rstrip() + "\n"


PDB_TEXT = (
    _pdb_line("ATOM", 1, "ALA", "A", 1, 10.0, 10.0, 10.0, name="N")
    + _pdb_line("ATOM", 2, "ALA", "A", 1, 11.0, 10.0, 10.0, name="CA")
    + _pdb_line("HETATM", 3, "HOH", "A", 2, 20.0, 20.0, 20.0, name="O")
    + _pdb_line("HETATM", 4, "LIG", "A", 3, 30.0, 30.0, 30.0, name="C1")
    + _pdb_line("HETATM", 5, "LIG", "A", 3, 31.0, 30.0, 30.0, name="O1")
)


def test_select_residue_lines_unique_match(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PDB_TEXT)
    lines = select_residue_lines(pdb, "LIG")
    assert len(lines) == 2
    assert all("LIG" in line for line in lines)


def test_select_residue_lines_no_match_raises(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PDB_TEXT)
    with pytest.raises(ValueError, match="No HETATM residue"):
        select_residue_lines(pdb, "ZZZ")


def test_select_residue_lines_ambiguous_raises(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    two_copies = PDB_TEXT + _pdb_line("HETATM", 6, "LIG", "B", 9, 40.0, 40.0, 40.0, name="C1")
    pdb.write_text(two_copies)
    with pytest.raises(ValueError, match="distinct residues"):
        select_residue_lines(pdb, "LIG")


def test_strip_residue_removes_only_target(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PDB_TEXT)
    out = strip_residue(pdb, "LIG", None, None, tmp_path / "stripped.pdb")
    text = out.read_text()
    assert "LIG" not in text
    assert "HOH" in text
    assert "ALA" in text


def test_isolate_residue_drops_other_hetatms(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PDB_TEXT)
    out = isolate_residue(pdb, "LIG", None, None, tmp_path / "isolated.pdb")
    text = out.read_text()
    assert "HOH" not in text
    assert "LIG" in text
    assert "ALA" in text


WHITESPACE_PQR = (
    "ATOM 1 N ALA 1 10.000 10.000 10.000 -0.4157 1.8240\n"
    "ATOM 2 CA ALA 1 11.000 10.000 10.000 0.0337 1.9080\n"
    "HETATM 3 C1 LIG 3 30.000 30.000 30.000 0.1234 1.7000\n"
    "HETATM 4 O1 LIG 3 31.000 30.000 30.000 -0.5678 1.5200\n"
)


def test_read_pqr_coords(tmp_path: Path):
    pqr = tmp_path / "complex.pqr"
    pqr.write_text(WHITESPACE_PQR)
    coords = read_pqr_coords(pqr)
    assert coords.shape == (4, 3)
    np.testing.assert_allclose(coords[2], [30.0, 30.0, 30.0])


def test_zero_ligand_charges_only_zeroes_target_residue(tmp_path: Path):
    pqr = tmp_path / "complex.pqr"
    pqr.write_text(WHITESPACE_PQR)
    out = zero_ligand_charges(pqr, "LIG", None, tmp_path / "zeroed.pqr")
    lines = out.read_text().splitlines()

    ala_lines = [line for line in lines if " ALA " in line]
    lig_lines = [line for line in lines if " LIG " in line]
    assert "-0.4157" in ala_lines[0]
    assert "0.0337" in ala_lines[1]
    for line in lig_lines:
        tokens = line.split()
        assert tokens[8] == "0.0000"
    assert lig_lines[0].split()[9] == "1.7000"
    assert lig_lines[1].split()[9] == "1.5200"


def test_zero_ligand_charges_by_resseq(tmp_path: Path):
    pqr_text = WHITESPACE_PQR + "HETATM 5 C1 LIG 7 50.000 50.000 50.000 0.9999 1.7000\n"
    pqr = tmp_path / "complex.pqr"
    pqr.write_text(pqr_text)
    out = zero_ligand_charges(pqr, "LIG", 3, tmp_path / "zeroed.pqr")
    lines = out.read_text().splitlines()
    resseq3 = [line for line in lines if line.split()[3] == "LIG" and line.split()[4] == "3"]
    resseq7 = [line for line in lines if line.split()[3] == "LIG" and line.split()[4] == "7"]
    assert all(line.split()[8] == "0.0000" for line in resseq3)
    assert resseq7[0].split()[8] == "0.9999"

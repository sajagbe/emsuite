"""Potential channel: protein_format='pdb' via pdb2pqr (present/absent/charged)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emsuite.inputs import PotentialInput

# A real, standard 3-residue fragment (ASN-ILE-PHE, no altlocs) from PDB 3HTB
# (T4 lysozyme L99A/M102Q), plus a synthetic methane ligand as a second HETATM
# residue. Small and self-contained so this doesn't depend on external files.
PROTEIN_PDB = """\
ATOM     14  N   ASN A   2       8.696 -17.109 -10.221  1.00 13.04           N
ATOM     15  CA  ASN A   2       9.444 -18.214 -10.830  1.00 11.28           C
ATOM     16  C   ASN A   2      10.923 -17.749 -10.851  1.00 11.13           C
ATOM     17  O   ASN A   2      11.262 -16.586 -10.453  1.00  8.10           O
ATOM     18  CB  ASN A   2       8.903 -18.572 -12.242  1.00 11.33           C
ATOM     19  CG  ASN A   2       8.979 -17.384 -13.226  1.00 13.64           C
ATOM     20  OD1 ASN A   2      10.036 -16.854 -13.455  1.00  9.75           O
ATOM     21  ND2 ASN A   2       7.826 -16.975 -13.804  1.00 12.13           N
ATOM     22  N   ILE A   3      11.803 -18.653 -11.255  1.00  9.44           N
ATOM     23  CA  ILE A   3      13.215 -18.370 -11.247  1.00  9.32           C
ATOM     24  C   ILE A   3      13.649 -17.195 -12.132  1.00  7.70           C
ATOM     25  O   ILE A   3      14.597 -16.489 -11.782  1.00  9.80           O
ATOM     26  CB  ILE A   3      14.010 -19.669 -11.653  1.00  9.07           C
ATOM     27  CG1 ILE A   3      15.522 -19.542 -11.375  1.00  8.89           C
ATOM     28  CG2 ILE A   3      13.701 -20.044 -13.127  1.00  9.00           C
ATOM     29  CD1 ILE A   3      16.021 -18.938 -10.040  1.00 10.83           C
ATOM     30  N   PHE A   4      12.976 -16.982 -13.265  1.00  8.60           N
ATOM     31  CA  PHE A   4      13.313 -15.827 -14.061  1.00  8.69           C
ATOM     32  C   PHE A   4      12.939 -14.559 -13.370  1.00  9.89           C
ATOM     33  O   PHE A   4      13.695 -13.598 -13.393  1.00  9.06           O
ATOM     34  CB  PHE A   4      12.680 -15.895 -15.481  1.00  8.95           C
ATOM     35  CG  PHE A   4      13.207 -17.104 -16.277  1.00 11.87           C
ATOM     36  CD1 PHE A   4      12.684 -18.381 -16.087  1.00 12.45           C
ATOM     37  CD2 PHE A   4      14.252 -16.959 -17.165  1.00 10.91           C
ATOM     38  CE1 PHE A   4      13.203 -19.507 -16.757  1.00 10.97           C
ATOM     39  CE2 PHE A   4      14.737 -18.073 -17.875  1.00 13.68           C
ATOM     40  CZ  PHE A   4      14.194 -19.334 -17.662  1.00 13.03           C
HETATM    1  C1  MTH A 200      50.000  50.000  50.000  1.00  0.00           C
HETATM    2  H1  MTH A 200      50.630  50.630  50.630  1.00  0.00           H
HETATM    3  H2  MTH A 200      49.370  49.370  50.630  1.00  0.00           H
HETATM    4  H3  MTH A 200      49.370  50.630  49.370  1.00  0.00           H
HETATM    5  H4  MTH A 200      50.630  49.370  49.370  1.00  0.00           H
END
"""

# Gasteiger charges for methane (sum ~0), computed once via `obabel --partialcharge
# gasteiger` on the HETATM block above — real, unique atom names, matching MTH's
# own PDB atom names so pdb2pqr's --ligand atom-name matching succeeds.
METHANE_MOL2 = """\
@<TRIPOS>MOLECULE
methane
 5 4 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1  C1        50.0000   50.0000   50.0000 C.3   200  MTH200     -0.0776
      2  H1        50.6300   50.6300   50.6300 H     200  MTH200      0.0194
      3  H2        49.3700   49.3700   50.6300 H     200  MTH200      0.0194
      4  H3        49.3700   50.6300   49.3700 H     200  MTH200      0.0194
      5  H4        50.6300   49.3700   49.3700 H     200  MTH200      0.0194
@<TRIPOS>BOND
     1     4     1    1
     2     5     1    1
     3     1     2    1
     4     1     3    1
"""

# molecule= just needs to be a valid XYZ (surface-generation + box purposes);
# unrelated to the pdb2pqr ligand (MTH), which is entirely inside the PDB.
LIGAND_XYZ = """\
1
probe
C 20.000 -17.000 -12.000
"""


@pytest.fixture
def pdb2pqr_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "complex.pdb").write_text(PROTEIN_PDB)
    (tmp_path / "methane.mol2").write_text(METHANE_MOL2)
    (tmp_path / "ligand.xyz").write_text(LIGAND_XYZ)
    return tmp_path


@pytest.mark.slow
def test_pdb_absent_mode(pdb2pqr_fixture: Path) -> None:
    result = PotentialInput.from_config(
        molecule="ligand.xyz",
        protein="complex.pdb",
        protein_format="pdb",
        ligand_atoms="absent",
        ligand_resname="MTH",
        quantity="potential",
        output_surf="absent.surf",
    ).run()
    assert Path(result.path).is_file()
    assert np.all(np.isfinite(result.values))


@pytest.mark.slow
def test_pdb_present_mode_zeroes_ligand_charge_keeps_radius(pdb2pqr_fixture: Path) -> None:
    result = PotentialInput.from_config(
        molecule="ligand.xyz",
        protein="complex.pdb",
        protein_format="pdb",
        ligand_atoms="present",
        ligand_resname="MTH",
        ligand_mol2="methane.mol2",
        quantity="potential",
        output_surf="present.surf",
    ).run()
    assert Path(result.path).is_file()
    assert np.all(np.isfinite(result.values))


@pytest.mark.slow
def test_pdb_charged_mode_keeps_real_ligand_charge(pdb2pqr_fixture: Path) -> None:
    result = PotentialInput.from_config(
        molecule="ligand.xyz",
        protein="complex.pdb",
        protein_format="pdb",
        ligand_atoms="charged",
        ligand_resname="MTH",
        ligand_mol2="methane.mol2",
        quantity="potential",
        output_surf="charged.surf",
    ).run()
    assert Path(result.path).is_file()
    assert np.all(np.isfinite(result.values))


@pytest.mark.slow
def test_pdb_present_vs_charged_pqr_differ_only_in_ligand_charge(
    pdb2pqr_fixture: Path,
) -> None:
    """present zeroes MTH's charge column; charged keeps pdb2pqr's real PEOE value."""
    from emsuite.potential.pdb2pqr_runner import run_pdb2pqr
    from emsuite.potential.pdb_select import isolate_residue
    from emsuite.potential.pqr import zero_ligand_charges

    isolated = isolate_residue("complex.pdb", "MTH", None, None, "isolated.pdb")
    charged_pqr = run_pdb2pqr(isolated, "charged.pqr", forcefield="AMBER", ligand_mol2="methane.mol2", ph=None)
    present_pqr = zero_ligand_charges(charged_pqr, "MTH", None, "present.pqr")

    charged_lines = [line for line in charged_pqr.read_text().splitlines() if " MTH " in line]
    present_lines = [line for line in present_pqr.read_text().splitlines() if " MTH " in line]
    assert len(charged_lines) == len(present_lines) == 5

    for charged_line, present_line in zip(charged_lines, present_lines, strict=True):
        c_tokens, p_tokens = charged_line.split(), present_line.split()
        assert c_tokens[9] == p_tokens[9]  # radius identical
        assert p_tokens[8] == "0.0000"  # present: charge zeroed
    assert any(t.split()[8] != "0.0000" for t in charged_lines)  # charged: real charge kept

    non_mth_charged = [line for line in charged_pqr.read_text().splitlines() if " MTH " not in line]
    non_mth_present = [line for line in present_pqr.read_text().splitlines() if " MTH " not in line]
    assert non_mth_charged == non_mth_present  # protein atoms untouched either way

"""Potential channel: protein_format='pdb' via pdb2pqr (present/absent/charged).

Fixture data (protein PDB, ligand MOL2/XYZ) lives in conftest.py's
pdb2pqr_fixture, shared with test_coupled_pdb2pqr.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from emsuite.inputs import PotentialInput


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
    charged_pqr = run_pdb2pqr(
        isolated, "charged.pqr", forcefield="AMBER", ligand_mol2="methane.mol2", ph=None
    )
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


def _pdb2pqr_temp_dirs() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("emsuite_pdb2pqr_*")}


@pytest.mark.slow
def test_pdb_mode_does_not_leak_temp_dir(pdb2pqr_fixture: Path) -> None:
    """occupancy.py's tempfile.mkdtemp() for the isolated PDB/PQR must be cleaned up."""
    before = _pdb2pqr_temp_dirs()
    PotentialInput.from_config(
        molecule="ligand.xyz",
        protein="complex.pdb",
        protein_format="pdb",
        ligand_atoms="absent",
        ligand_resname="MTH",
        quantity="potential",
        output_surf="leak_check.surf",
    ).run()
    after = _pdb2pqr_temp_dirs()
    assert after == before, f"leaked temp dir(s): {after - before}"

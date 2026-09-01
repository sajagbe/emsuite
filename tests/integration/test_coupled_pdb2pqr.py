"""Coupled channel drives the pdb2pqr PDB path through to tuning.

Fixture data (protein PDB, ligand MOL2/XYZ) lives in conftest.py's
pdb2pqr_fixture, shared with test_potential_pdb2pqr.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite.inputs import CoupledInput


@pytest.mark.slow
def test_coupled_absent_mode_via_pdb2pqr(pdb2pqr_fixture: Path) -> None:
    result = CoupledInput.from_config(
        molecule="ligand.xyz",
        protein="complex.pdb",
        protein_format="pdb",
        ligand_atoms="absent",
        ligand_resname="MTH",
        potential_quantity="potential",
        properties=["homo"],
        basis_set="sto-3g",
        parallel=False,
        output_surf="coupled_absent.surf",
    ).run()
    assert result.potential.path
    assert Path(result.potential.path).is_file()
    assert result.tuning.results_dir
    assert Path(result.tuning.results_dir).is_dir()


@pytest.mark.slow
def test_coupled_charged_mode_via_pdb2pqr(pdb2pqr_fixture: Path) -> None:
    result = CoupledInput.from_config(
        molecule="ligand.xyz",
        protein="complex.pdb",
        protein_format="pdb",
        ligand_atoms="charged",
        ligand_resname="MTH",
        ligand_mol2="methane.mol2",
        potential_quantity="potential",
        properties=["homo"],
        basis_set="sto-3g",
        parallel=False,
        output_surf="coupled_charged.surf",
    ).run()
    assert result.potential.path
    assert Path(result.potential.path).is_file()
    assert result.tuning.results_dir
    assert Path(result.tuning.results_dir).is_dir()

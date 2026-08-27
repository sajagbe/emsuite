"""Ligand occupancy PQR construction (AE5–AE6, KTD4)."""

import numpy as np
import pytest

from emsuite.config import ConfigValidationError, validate_potential_params
from emsuite.geometry import Geometry
from emsuite.potential.occupancy import assemble_pqr
from emsuite.potential.pqr import write_pqr


def _toy_pair():
    protein = Geometry(symbols=("C", "O"), coords=np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))
    ligand = Geometry(symbols=("H",), coords=np.array([[20.0, 0.0, 0.0]]))
    protein_charges = np.array([0.5, -0.5])
    return protein, ligand, protein_charges


def test_present_pqr_includes_ligand_at_zero_charge(tmp_path):
    protein, ligand, protein_charges = _toy_pair()
    atoms, charges, box = assemble_pqr(ligand, protein, protein_charges, "present")
    assert len(atoms) == 3
    assert atoms[-1][0] == "H"
    assert charges[-1] == 0.0
    assert charges[0] == pytest.approx(0.5)
    path = write_pqr(atoms, charges.tolist(), tmp_path / "present.pqr")
    text = path.read_text()
    assert text.count("\n") == 3
    assert "  0.0000" in text.splitlines()[-1]


def test_absent_pqr_omits_ligand_but_box_spans_ligand():
    protein, ligand, protein_charges = _toy_pair()
    atoms, charges, box = assemble_pqr(ligand, protein, protein_charges, "absent")
    assert len(atoms) == 2
    assert all(symbol != "H" for symbol, *_ in atoms)
    assert box[:, 0].max() == pytest.approx(20.0)
    assert len(charges) == 2


def test_ligand_atoms_absent_requires_protein():
    with pytest.raises(ConfigValidationError, match="requires protein"):
        validate_potential_params({"molecule": "lig.xyz", "ligand_atoms": "absent"})


def test_ligand_alias_fills_molecule():
    params = validate_potential_params({"ligand": "lig.xyz"})
    assert params["molecule"] == "lig.xyz"
    assert params["ligand"] == "lig.xyz"
    assert params["ligand_atoms"] == "present"

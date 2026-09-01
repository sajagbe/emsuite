"""Config schema validation tests."""

import pytest

from emsuite.config import ConfigValidationError, validate_surface_params, validate_tuning_params
from emsuite.config.schemas import validate_coupled_params, validate_potential_params


def test_validate_surface_params_requires_input():
    with pytest.raises(ConfigValidationError):
        validate_surface_params({"input_type": None, "input_data": "C"})


def test_validate_tuning_params_accepts_molecule_alias():
    params = validate_tuning_params(
        {"xyz_file": "mol.xyz", "surface_file": "mol.surf", "properties": ["homo"]}
    )
    assert params["molecule"] == "mol.xyz"


def test_validate_coupled_params_skips_potential_validation_with_potential_surf():
    # No protein/ligand_atoms/method/quantity given — would fail validate_potential_params
    # if it ran, since potential_surf means potential is never actually computed.
    params = validate_coupled_params(
        {
            "molecule": "m.xyz",
            "properties": ["homo"],
            "potential_surf": "precomputed.surf",
        }
    )
    assert params["potential_surf"] == "precomputed.surf"


def test_validate_coupled_params_still_validates_potential_without_potential_surf():
    with pytest.raises(ConfigValidationError, match="requires protein"):
        validate_coupled_params(
            {
                "molecule": "m.xyz",
                "properties": ["homo"],
                "ligand_atoms": "absent",
            }
        )


def test_ligand_atoms_charged_requires_pdb_protein_format():
    # 'charged' only means anything on the pdb2pqr path — without it, this used to
    # pass validation and fail later with a confusing generic ValueError from
    # assemble_pqr instead of a clear ConfigValidationError here.
    with pytest.raises(ConfigValidationError, match="protein_format='pdb'"):
        validate_potential_params({"molecule": "m.xyz", "ligand_atoms": "charged"})


def test_ligand_atoms_charged_accepted_with_pdb_protein_format():
    params = validate_potential_params(
        {
            "molecule": "m.xyz",
            "protein": "complex.pdb",
            "protein_format": "pdb",
            "ligand_atoms": "charged",
            "ligand_resname": "LIG",
            "ligand_mol2": "lig.mol2",
        }
    )
    assert params["ligand_atoms"] == "charged"
    assert params["protein_format"] == "pdb"

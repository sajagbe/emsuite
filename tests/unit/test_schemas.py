"""Config schema validation tests."""

import pytest

from emsuite.config import ConfigValidationError, validate_surface_params, validate_tuning_params
from emsuite.config.schemas import validate_coupled_params


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

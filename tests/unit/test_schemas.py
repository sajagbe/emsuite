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
    # No pdb/surface_file given: fine, because potential_surf means the potential
    # is never computed.
    params = validate_coupled_params(
        {
            "molecule": "m.xyz",
            "properties": ["homo"],
            "potential_surf": "precomputed.surf",
        }
    )
    assert params["potential_surf"] == "precomputed.surf"


def test_validate_coupled_params_requires_pdb_and_allows_automatic_surface():
    with pytest.raises(ConfigValidationError, match="pdb"):
        validate_coupled_params({"molecule": "m.xyz", "properties": ["homo"]})
    params = validate_coupled_params({"molecule": "m.xyz", "properties": ["homo"], "pdb": "c.pdb"})
    assert params["surface_density"] == 0.5


def test_validate_coupled_params_checks_potential_settings():
    with pytest.raises(ConfigValidationError, match="quantity"):
        validate_coupled_params(
            {
                "molecule": "m.xyz",
                "properties": ["homo"],
                "pdb": "c.pdb",
                "surface_file": "m.surf",
                "potential_quantity": "field",
            }
        )

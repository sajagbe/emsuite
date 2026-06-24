"""Config schema validation tests."""

import pytest

from emsuite.config import ConfigValidationError, validate_surface_params, validate_tuning_params


def test_validate_surface_params_requires_input():
    with pytest.raises(ConfigValidationError):
        validate_surface_params({"input_type": None, "input_data": "C"})


def test_validate_tuning_params_accepts_molecule_alias():
    params = validate_tuning_params(
        {"xyz_file": "mol.xyz", "surface_file": "mol.surf", "properties": ["homo"]}
    )
    assert params["molecule"] == "mol.xyz"

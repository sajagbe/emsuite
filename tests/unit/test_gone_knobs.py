"""Gone-knob tests: removed methods and properties must be rejected (AE10)."""

import pytest

from emsuite.config import ConfigValidationError, validate_potential_params, validate_tuning_params
from emsuite.tuning.properties import PROPERTY_CONFIG, setup_calculation


def test_coulomb_method_rejected():
    with pytest.raises(ConfigValidationError, match="coulomb"):
        validate_potential_params({"molecule": "m.xyz", "method": "coulomb"})


def test_bond_scan_atoms_rejected():
    with pytest.raises(ConfigValidationError, match="bond_scan_atoms"):
        validate_potential_params({"molecule": "m.xyz", "bond_scan_atoms": [0, 1]})


def test_stark_gap_rejected_by_schema():
    with pytest.raises(ConfigValidationError, match="stark_gap"):
        validate_tuning_params(
            {"molecule": "m.xyz", "surface_file": "s.surf", "properties": ["stark_gap"]}
        )


def test_stark_gap_not_in_registry():
    assert "stark_gap" not in PROPERTY_CONFIG
    with pytest.raises(KeyError, match="stark_gap"):
        setup_calculation(["stark_gap"])

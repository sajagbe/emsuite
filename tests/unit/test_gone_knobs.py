"""Gone-knob tests: removed methods and properties must be rejected (AE10)."""

import pytest

from emsuite.config import ConfigValidationError, validate_tuning_params
from emsuite.potential.spec import PotentialSpec
from emsuite.tuning.properties import PROPERTY_CONFIG, setup_calculation


def test_coulomb_method_rejected():
    with pytest.raises(ConfigValidationError, match="coulomb"):
        PotentialSpec(pdb="c.pdb", surface_file="m.surf", method="coulomb")


def test_bond_scan_atoms_rejected(tmp_path):
    cfg = tmp_path / "potential.in"
    cfg.write_text("pdb = 'c.pdb'\nsurface_file = 'm.surf'\nbond_scan_atoms = [0, 1]\n")
    with pytest.raises(ConfigValidationError, match="bond_scan_atoms"):
        PotentialSpec.from_file(cfg, check_files=False)


@pytest.mark.parametrize(
    "line",
    ["molecule = 'm.xyz'", "protein = 'p.xyz'", "protein_format = 'pdb'", "ligand_mol2 = 'l.mol2'"],
)
def test_old_potential_keys_rejected(tmp_path, line):
    cfg = tmp_path / "potential.in"
    cfg.write_text(f"pdb = 'c.pdb'\nsurface_file = 'm.surf'\n{line}\n")
    with pytest.raises(ConfigValidationError, match="no longer a potential input key"):
        PotentialSpec.from_file(cfg, check_files=False)


def test_old_coupled_keys_rejected(tmp_path):
    from emsuite.inputs import CoupledInput

    cfg = tmp_path / "coupled.in"
    cfg.write_text("molecule = 'm.xyz'\nproperties = ['homo']\nligand_atoms = 'present'\n")
    with pytest.raises(ConfigValidationError, match="no longer a coupled input key"):
        CoupledInput.from_file(cfg)


def test_stark_gap_rejected_by_schema():
    with pytest.raises(ConfigValidationError, match="stark_gap"):
        validate_tuning_params(
            {"molecule": "m.xyz", "surface_file": "s.surf", "properties": ["stark_gap"]}
        )


def test_stark_gap_not_in_registry():
    assert "stark_gap" not in PROPERTY_CONFIG
    with pytest.raises(KeyError, match="stark_gap"):
        setup_calculation(["stark_gap"])


def test_fukui_rejected_by_schema():
    with pytest.raises(ConfigValidationError, match="fukui_plus"):
        validate_tuning_params(
            {"molecule": "m.xyz", "surface_file": "s.surf", "properties": ["fukui_plus"]}
        )
    with pytest.raises(ConfigValidationError, match="fukui_minus"):
        validate_tuning_params(
            {"molecule": "m.xyz", "surface_file": "s.surf", "properties": ["fukui_minus"]}
        )


def test_fukui_not_in_registry():
    assert "fukui_plus" not in PROPERTY_CONFIG
    assert "fukui_minus" not in PROPERTY_CONFIG
    with pytest.raises(KeyError, match="fukui_plus"):
        setup_calculation(["fukui_plus"])
    with pytest.raises(KeyError, match="fukui_minus"):
        setup_calculation(["fukui_minus"])

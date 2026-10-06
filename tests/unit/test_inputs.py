"""Input objects: file and constructor paths share validation."""

from pathlib import Path

import pytest

from emsuite.config.schemas import ConfigValidationError
from emsuite.inputs import PotentialInput, SurfaceInput, TuningInput


def test_surface_input_file_matches_constructor(tmp_path: Path):
    cfg = tmp_path / "surface.in"
    cfg.write_text("input_type = 'SMILES'\ninput_data = 'CCO'\n")
    from_file = SurfaceInput.from_file(cfg)
    from_ctor = SurfaceInput(input_type="SMILES", input_data="CCO")
    assert from_file == from_ctor


def test_potential_input_file_matches_constructor(tmp_path: Path):
    cfg = tmp_path / "potential.in"
    cfg.write_text("molecule = 'm.xyz'\nquantity = 'charge'\n")
    from_file = PotentialInput.from_file(cfg)
    from_ctor = PotentialInput(molecule="m.xyz", quantity="charge")
    assert from_file == from_ctor
    assert from_file.method == "apbs"
    assert from_ctor.ligand == "m.xyz"


def test_tuning_input_file_matches_constructor(tmp_path: Path):
    cfg = tmp_path / "tuning.in"
    cfg.write_text("molecule = 'm.xyz'\nsurface_file = 'm.surf'\nproperties = ['homo', 'gap']\n")
    from_file = TuningInput.from_file(cfg)
    from_ctor = TuningInput(molecule="m.xyz", surface_file="m.surf", properties=("homo", "gap"))
    assert from_file == from_ctor
    assert from_file.properties == ("homo", "gap")


def test_surface_constructor_rejects_missing_required():
    with pytest.raises(ConfigValidationError, match="input_type|input_data|missing"):
        SurfaceInput(input_type="", input_data="C")


def test_potential_constructor_normalizes_method_case():
    inp = PotentialInput(molecule="m.xyz", method="APBS", quantity="CHARGE")
    assert inp.method == "apbs"
    assert inp.quantity == "charge"


def test_coupled_to_potential_and_tuning_input():
    from emsuite.inputs import CoupledInput

    coupled = CoupledInput(
        molecule="m.xyz",
        surface_file="m.surf",
        potential_quantity="charge",
        properties=("homo",),
        parallel=False,
        basis_set="sto-3g",
    )
    pot = coupled.to_potential_input()
    assert pot.molecule == "m.xyz"
    assert pot.quantity == "charge"
    assert pot.method == "apbs"
    tune = coupled.to_tuning_input("out.surf")
    assert tune.surface_file == "out.surf"
    assert tune.properties == ("homo",)
    assert tune.parallel is False
    assert tune.basis_set == "sto-3g"

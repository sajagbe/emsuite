"""Input objects from kwargs and .in files are equal."""

from pathlib import Path

from emsuite.inputs import PotentialInput, SurfaceInput, TuningInput


def test_surface_input_file_matches_kwargs(tmp_path: Path):
    cfg = tmp_path / "surface.in"
    cfg.write_text("input_type = 'SMILES'\ninput_data = 'CCO'\n")
    from_file = SurfaceInput.from_file(cfg)
    from_kwargs = SurfaceInput.from_config(input_type="SMILES", input_data="CCO")
    assert from_file == from_kwargs


def test_potential_input_file_matches_kwargs(tmp_path: Path):
    cfg = tmp_path / "potential.in"
    cfg.write_text("molecule = 'm.xyz'\nquantity = 'charge'\n")
    from_file = PotentialInput.from_file(cfg)
    from_kwargs = PotentialInput.from_config(molecule="m.xyz", quantity="charge")
    assert from_file == from_kwargs
    assert from_file.method == "apbs"


def test_tuning_input_file_matches_kwargs(tmp_path: Path):
    cfg = tmp_path / "tuning.in"
    cfg.write_text("molecule = 'm.xyz'\nsurface_file = 'm.surf'\nproperties = ['homo', 'gap']\n")
    from_file = TuningInput.from_file(cfg)
    from_kwargs = TuningInput.from_config(
        molecule="m.xyz", surface_file="m.surf", properties=["homo", "gap"]
    )
    assert from_file == from_kwargs
    assert from_file.properties == ("homo", "gap")
